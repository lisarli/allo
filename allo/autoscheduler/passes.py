# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module
import os


from .._mlir.ir import (
    Location,
    InsertionPoint,
    IntegerSetAttr,
    Block,
    Module,
    StringAttr,
    UnitAttr,
    IntegerAttr,
    IntegerType,
    WalkResult,
    Operation,
    MemRefType,
    BlockArgument
)
from .._mlir.dialects import (
    func as func_d,
    affine as affine_d,
    memref as memref_d,
)
from .._mlir.passmanager import PassManager as mlir_pass_manager
from ..customize import Partition, Schedule
from ..ir.transform import find_func_in_module
from ..ir.utils import MockBuffer
from ..verify import verify


from .util import (
    check_perfect_affine_kernel,
    check_call_graph_acyclic,
    check_all_functions_inlined,
)

from .dfg import DFG, DFGNodeType, NodeInfo, DFGAnalysisResult, LoopInfo
from .primitives import SchedulePrimitive, UnresolvedFIFOPrimitive
from .config import AutoschedulerConfig

DEBUG_POINTS = [
    "mlir_preprocess",
    "dataflow_canonicalization",
    "outline_loops",
    "loop_opts",
    None,
]
PARALLELISM_MODELS = ["graph", "node", "combined"]


def dataflow_optimization_pass(
    schedule: Schedule,
    cfg: AutoschedulerConfig,
) -> Schedule:
    """
    Applies autoscheduler optimization passes to the schedule.
    """
    assert (
        cfg.debug_point is None or cfg.debug_point in DEBUG_POINTS
    ), f"Invalid debug point: {cfg.debug_point}"
    assert (
        cfg.kind is None or cfg.kind in PARALLELISM_MODELS
    ), f"Invalid parallelism model: {cfg.kind}"

    assert check_call_graph_acyclic(schedule.module), "Call graph is not acyclic"
    top_fn_name = schedule.top_func.name.value
    mod = _mlir_preprocess(schedule.module, top_fn_name)
    if cfg.debug_point == "mlir_preprocess":
        return Schedule(
            mod,
            find_func_in_module(mod, top_fn_name),
            schedule.func_args,
            schedule.ip,
            schedule.ext_libs,
            schedule.inst_list,
        )
    assert check_all_functions_inlined(
        mod, top_fn_name
    ), "All functions are not inlined"
    assert check_perfect_affine_kernel(
        mod
    ), "Input kernel is not a perfect affine kernel"

    # Dataflow canonicalization pass
    try:
        mod_dcp = _dataflow_canonicalization_pass(mod)
    except Exception as e:
        print("Error: failed to run dataflow canonicalization pass, printing module...")
        print(mod)
        raise e
    if cfg.debug_point == "dataflow_canonicalization":
        return Schedule(
            mod_dcp,
            find_func_in_module(mod_dcp, top_fn_name),
            schedule.func_args,
            schedule.ip,
            schedule.ext_libs,
            schedule.inst_list,
        )

    dfg = DFG.from_module(mod_dcp, cfg.dsp_factors, cfg.mem_w_ports, cfg.mem_r_ports)

    # name all unnamed buffers
    mod_dcp = name_buffers_pass(mod_dcp)

    # build performance model
    match cfg.kind:
        case "graph":
            result: DFGAnalysisResult = dfg.create_performance_model(
                enable_tile=False,
                verbose=cfg.verbose,
                dsp_limit=cfg.dsp_limit,
                tiling_limit=cfg.tiling_limit,
            )

        case "node":
            # solve seperately for a fixed permutation
            permutation_result: DFGAnalysisResult = dfg.create_performance_model(
                enable_tile=False,
                verbose=cfg.verbose,
                dsp_limit=cfg.dsp_limit,
                tiling_limit=cfg.tiling_limit,
            )

            # fix the permutation to the previously solved solution and solve for tiling factors
            result: DFGAnalysisResult = dfg.create_performance_model(
                permutation_result.loop_permutations,
                enable_tile=True,
                verbose=cfg.verbose,
                dsp_limit=cfg.dsp_limit,
                tiling_limit=cfg.tiling_limit,
            )

        case "combined":
            result: DFGAnalysisResult = dfg.create_performance_model(
                enable_tile=True,
                verbose=cfg.verbose,
                dsp_limit=cfg.dsp_limit,
                tiling_limit=cfg.tiling_limit,
            )
        case _:
            raise ValueError(f"Invalid parallelism model: {cfg.kind}")

    # extract FIFO primitives prior to outlining, since this requires IR manipulation
    # dependent on references from the inlined result
    print(f"[debug] Top func args for {top_fn_name}:")
    for i, a in enumerate(schedule.func_args.get(top_fn_name, [])):
        print(f"  index={i}, arg={a}, type={type(a)}")
    array_parts = extract_array_partitions(result, dfg, top_fn_name, schedule)
    fifos = extract_buffer_to_fifo(result, dfg, schedule, top_fn_name)
    # Build a set of buffer names that will become FIFOs
    fifo_bufs = set()
    for p in fifos:
        if isinstance(p, UnresolvedFIFOPrimitive):
            fifo_bufs.add(p.buffer_name)
        else:
            # SchedulePrimitive.to: args[0] is MockBuffer(func,name)
            if p.kind == "to" and isinstance(p.args[0], MockBuffer):
                fifo_bufs.add(p.args[0].name)
    # Filter array partitions to avoid streams/FIFOs
    array_parts = [
        ap for ap in array_parts
        if not (isinstance(ap.args[0], MockBuffer) and ap.args[0].name in fifo_bufs)
    ]
    print("Number of array partitions after removing duplicate FIFOs:", len(array_parts))
    for primitive in array_parts:
        print(f"\t{primitive}")

    mod_outlined, node_to_fn = outline_loops_pass(schedule.module, dfg)
    # construct the schedule with the outlined module
    schedule = Schedule(
        mod_outlined,
        find_func_in_module(mod_outlined, top_fn_name),
        schedule.func_args,
        schedule.ip,
        schedule.ext_libs,
        schedule.inst_list,
    )
    # Seed func_args for all outlined functions with the correct arity.
    for func in schedule.module.body.operations:
        if isinstance(func, func_d.FuncOp):
            fname = func.name.value
            n_args = len(func.arguments)  # number of BlockArguments
            current = schedule.func_args.get(fname)
            if current is None:
                # populate with placeholders "arg0", "arg1", ...
                schedule.func_args[fname] = [f"arg{i}" for i in range(n_args)]
            elif len(current) < n_args:
                schedule.func_args[fname].extend(f"arg{i}" for i in range(len(current), n_args))

    if cfg.debug_point == "outline_loops":
        return schedule

    # clone the original schedule for verification
    try:
        import past  # pylint: disable=unused-import
    except ImportError:
        cfg.verify = False

    if cfg.verify:
        # hacky clone
        mod_outlined_clone = Module.parse(
            mod_outlined.operation.get_asm(), mod_outlined.context
        )
        original_schedule = Schedule(
            mod_outlined_clone,
            find_func_in_module(mod_outlined_clone, top_fn_name),
            schedule.func_args,
            schedule.ip,
            schedule.ext_libs,
            schedule.inst_list,
        )

    # loop opt extraction post-outlining
    loop_opts = extract_reorder_and_pipeline(result, dfg, node_to_fn, schedule)
    loop_tiling = extract_tiling(result, node_to_fn, schedule)

    for primitive in loop_opts:
        primitive.applyTo(schedule)

    for primitive in loop_tiling:
        primitive.applyTo(schedule)
    post_process = _tiling_post_process(schedule)

    for primitive in post_process:
        primitive.applyTo(schedule)

    for primitive in array_parts:
        print("applying array paritition primitive,", primitive)
        primitive.applyTo(schedule)
    # Ensure callsites match callee memref param layouts after partitioning
    _fix_call_memref_layout_mismatches(schedule.module)

    if cfg.verbose:
        print("Loop opts:")
        for primitive in loop_opts:
            print(f"\t{primitive}")
        print("Loop tiling:")
        for primitive in loop_tiling:
            print(f"\t{primitive}")
        if array_parts:
            print("Array partitions:")
            for primitive in array_parts:
                print(f"\t{primitive}")

    if cfg.debug_point == "loop_opts":
        return schedule

    if cfg.verbose:
        print("FIFOs:")
        for primitive in fifos:
            print(f"\t{primitive}")

    if cfg.verify:
        verifier = verify(schedule, original_schedule)
        assert (
            verifier
        ), "Failed verification: Schedule is not equivalent to original schedule"

    _canonicalize(schedule)

    for primitive in fifos:
        if isinstance(primitive, UnresolvedFIFOPrimitive):
            primitive = primitive.resolve(top_fn_name, node_to_fn)
        primitive.applyTo(schedule)

    return schedule


def _mlir_preprocess(module, top_func_name):
    """
    Performs linalg-to-affine lowering, then aggressive inlining on an MLIR module, then removes all (dead) functions except the top-level function.
    """
    # Configure for maximum inlining - no recursion limit and always inline
    MAX_ITER = INLINE_THRESHOLD = 999999
    pipeline = (
        f"builtin.module("
        f"convert-linalg-to-affine-loops,"
        f"inline{{max-iterations={MAX_ITER} inlining-threshold={INLINE_THRESHOLD}}},"
        f"symbol-privatize{{exclude={top_func_name}}},"
        f"symbol-dce,"
        f"func.func(affine-scalrep)"
        f")"
    )
    try:
        with module.context:
            mlir_pass_manager.parse(pipeline).run(module.operation)
        return module
    except Exception as e:
        print("Error: failed to run MLIR passes, printing module...")
        print(module)
        raise e


def _canonicalize(schedule: Schedule) -> Schedule:
    pipeline = "builtin.module(canonicalize)"
    try:
        with schedule.module.context:
            mlir_pass_manager.parse(pipeline).run(schedule.module.operation)
        return schedule
    except Exception as e:
        print("Error: failed to run MLIR passes, printing module...")
        print(schedule.module)
        raise e


def _dataflow_canonicalization_pass(module):
    """
    Implements the dataflow canonicalization pass as described in the Stream-HLS paper (https://arxiv.org/pdf/2501.09118)

    This pass ensures that the program is compatible with dataflow architectures by transforming shared buffers to adhere to single-producer-single-consumer patterns. This pass does not handle complex patterns involving multiple producers writing to the same buffer, except in the case of reduction loops.
    """
    with module.context, Location.unknown():
        for op in module.body.operations:
            if not isinstance(op, func_d.FuncOp):
                continue
            canonicalize_fn(op)
    return module


def canonicalize_fn(op: func_d.FuncOp):
    ops = list(op.entry_block.operations)
    for op_in_block in ops:
        if isinstance(op_in_block, memref_d.AllocOp):
            canonicalize_alloc(op_in_block)


def canonicalize_alloc(alloc_op):
    loads = []  # (op, idx)
    stores = []  # ops
    ret = []
    for use in alloc_op.result.uses:
        user = use.owner
        if isinstance(user, (memref_d.LoadOp, affine_d.AffineLoadOp, func_d.CallOp)):
            for idx, operand in enumerate(user.operands):
                if operand == alloc_op.result:
                    loads.append((user, idx))
        elif isinstance(user, (memref_d.StoreOp, affine_d.AffineStoreOp)):
            stores.append(user)
        elif isinstance(user, func_d.ReturnOp):
            ret.append(user)
    memref_type = alloc_op.result.type
    shape = memref_type.shape
    orig_name = (
        alloc_op.attributes["name"].value if "name" in alloc_op.attributes else "buffer"
    )
    if len(shape) == 0:
        # should constants be propogated?
        return

    # single store with multiple loads.
    if len(stores) == 1 and len(loads) > 1:
        store = stores[0]
        for i, (load, idx) in enumerate(loads[1:]):
            new_alloc = alloc_op.operation.clone(ip=InsertionPoint(alloc_op))
            name = f"{orig_name}_split_{i}"
            new_alloc.attributes["name"] = StringAttr.get(name)
            store_dup = store.clone(ip=InsertionPoint(store))
            store_dup.operation.replace_uses_of_with(alloc_op.result, new_alloc.result)
            store_dup.attributes["from"] = StringAttr.get(name)
            load.operation.operands[idx] = new_alloc.result
        return

    # store-load-store-load loop redution pattern
    if len(stores) == 2 and (len(loads) == 2 or len(loads) == 1 and len(ret) == 1):
        l_ops = [l[0] for l in loads]
        if store_load_store_load_pattern(alloc_op, l_ops, stores, ret):
            return

    # multiple loads and multiple stores
    if len(stores) >= 2 and len(loads) >= 1:
        raise NotImplementedError(
            f"Complex pattern detected in alloc op {alloc_op}; additional canonicalization not implemented yet."
        )


def store_load_store_load_pattern(alloc_op, loads, stores, ret):
    """
    Transforms reduction loops to satisfy the condition that the number of writes to a shared buffer equals the number of reads.
    """
    assert len(loads) + len(ret) == 2 and len(stores) == 2

    loop_load, loop_store = None, None

    # find loop_load and loop_store
    for load in loads:
        for store in stores:
            if load.parent == store.parent:
                loop_load = load
                loop_store = store
                break
        if loop_load:
            break

    if not loop_load or not loop_store:
        return False

    store_op = [s for s in stores if s != loop_store][0]

    load_op = [l for l in loads if l != loop_load][0] if len(ret) == 0 else ret[0]

    # check for unsupported store_op in an if block
    parent = store_op.parent
    while parent:
        if isinstance(parent, affine_d.AffineIfOp):
            return False
        parent = parent.parent

    loop_nest = []
    current_op = loop_load.parent
    while current_op:
        if isinstance(current_op.opview, affine_d.AffineForOp):
            loop_nest.append(current_op.opview)
        current_op = current_op.parent
    if not loop_nest:
        return False

    ip = InsertionPoint(alloc_op)
    memref_type = alloc_op.result.type
    new_alloc1 = memref_d.AllocOp(memref_type, [], [], ip=ip)
    new_alloc2 = memref_d.AllocOp(memref_type, [], [], ip=ip)

    irrelevant_loops = [
        loop for loop in loop_nest if loop.induction_variable not in loop_load.indices
    ]

    irrelevant_ivs = [loop.opview.induction_variable for loop in irrelevant_loops]

    if len(irrelevant_ivs) == 0:
        return False

    with InsertionPoint.at_block_begin(loop_nest[0].body):
        first_iter_set = affine_d.IntegerSet.get(
            len(irrelevant_ivs),
            0,
            [affine_d.AffineExpr.get_dim(i) for i in range(len(irrelevant_ivs))],
            [True] * len(irrelevant_ivs),
        )

        first_iter_if = affine_d.AffineIfOp(
            results_=[], _gen_arg_0=irrelevant_ivs, loc=Location.unknown()
        )

        first_iter_if.attributes["condition"] = IntegerSetAttr.get(first_iter_set)

    # In the if block, load from original buffer and store to new_alloc1
    then_block = Block.create_at_start(parent=first_iter_if.thenRegion)
    with InsertionPoint(then_block):
        new_load = affine_d.AffineLoadOp(
            memref_type.element_type, alloc_op.result, loop_load.indices, loop_load.map
        )
        affine_d.AffineStoreOp(
            new_load.result, new_alloc1.result, loop_load.indices, loop_load.map
        )
        affine_d.AffineYieldOp([])

    loop_load.operation.replace_uses_of_with(alloc_op.result, new_alloc1.result)
    loop_store.operation.replace_uses_of_with(alloc_op.result, new_alloc1.result)

    upper_bounds = [loop.upperBoundMap.value.results[0] for loop in irrelevant_loops]
    last_iter_set = affine_d.IntegerSet.get(
        len(irrelevant_ivs),
        0,
        [
            affine_d.AffineExpr.get_dim(i) - upper_bound + 1
            for i, upper_bound in enumerate(upper_bounds)
        ],
        [True] * len(irrelevant_ivs),
    )

    last_iter_if = affine_d.AffineIfOp(
        results_=[], _gen_arg_0=irrelevant_ivs, loc=Location.unknown(), ip=ip
    )

    last_iter_if.move_after(loop_store)

    last_iter_if.attributes["condition"] = IntegerSetAttr.get(last_iter_set)

    final_then_block = Block.create_at_start(parent=last_iter_if.thenRegion)
    with InsertionPoint(final_then_block):
        final_loop_load = affine_d.AffineLoadOp(
            memref_type.element_type,
            new_alloc1.result,
            loop_load.indices,
            loop_load.map,
        )
        affine_d.AffineStoreOp(
            final_loop_load.result, new_alloc2.result, loop_load.indices, loop_load.map
        )
        affine_d.AffineYieldOp([])

    load_op.operation.replace_uses_of_with(alloc_op.result, new_alloc2.result)

    return True


def name_buffers_pass(module: Module):
    unnamed_ct = 0

    def name_buffer_helper(op):
        nonlocal unnamed_ct
        if isinstance(op.opview, memref_d.AllocOp):
            if "name" not in op.attributes:
                buffer_name = f"_buffer_{unnamed_ct}"
                op.attributes["name"] = StringAttr.get(buffer_name)
                unnamed_ct += 1
            else:
                buffer_name = op.attributes["name"].value
            for use in op.result.uses:
                if isinstance(use.owner, (memref_d.LoadOp, affine_d.AffineLoadOp)):
                    use.owner.attributes["from"] = StringAttr.get(buffer_name)
                elif isinstance(use.owner, (memref_d.StoreOp, affine_d.AffineStoreOp)):
                    use.owner.attributes["to"] = StringAttr.get(buffer_name)

        return WalkResult(0)

    with module.context:
        module.operation.walk(name_buffer_helper)
        return module


def outline_loops_pass(
    module: Module, dfg: DFG = None
) -> tuple[Module, dict[int, str]]:
    with module.context:
        for func in module.body.operations:
            if not isinstance(func, func_d.FuncOp):
                continue
            for op in func.body.blocks[0]:
                if isinstance(op, affine_d.AffineForOp):
                    op.attributes["top_level"] = UnitAttr.get()
        if dfg:
            for node_id in dfg.nodes:
                node = dfg.nodes[node_id]
                if node.type == DFGNodeType.AFFINE:
                    node.op.attributes["node_id"] = IntegerAttr.get(
                        IntegerType.get_unsigned(32), node_id
                    )

    module_content = module.operation.get_asm()

    with open(
        os.path.join(os.path.dirname(__file__), "outline_loops.mlir"),
        "r",
        encoding="utf-8",
    ) as f:
        transform_content = f.read()

    combined_content = f"{module_content}\n{transform_content}"

    with module.context as ctx:
        ctx.allow_unregistered_dialects = True
        try:
            combined_module = Module.parse(combined_content, ctx)
            pipeline = "builtin.module(transform-interpreter{entry-point=outline_affine_loops})"
            mlir_pass_manager.parse(pipeline).run(combined_module.operation)
            processed_module, node_to_fn_map = post_process_module(
                Module.parse(
                    combined_module.operation.regions[0]
                    .blocks[0]
                    .operations[0]
                    .get_asm(),
                    ctx,
                )
            )
            return processed_module, node_to_fn_map

        except Exception as e:
            print("Error: failed to run MLIR passes, printing module...")
            print(combined_content)
            raise e


def post_process_module(module: Module) -> tuple[Module, dict[int, str]]:
    """
    Post-processes a module by adding loop names and operation names
    and builds a mapping from node IDs to function names.

    Args:
        module: The MLIR module to process.

    Returns:
        the processed module and a dictionary mapping node IDs to function names.
    """
    loop_counter = 0
    node_to_fn = {}

    def process_op(op, func_name):
        nonlocal loop_counter

        if isinstance(op, affine_d.AffineForOp):
            if "loop_name" not in op.attributes:
                op.attributes["loop_name"] = StringAttr.get(f"L_{loop_counter}")
                loop_counter += 1

            if "top_level" in op.attributes:
                assert "node_id" in op.attributes
                node_id = int(op.attributes["node_id"].value)

                node_to_fn[node_id] = func_name

                if "op_name" not in op.attributes:
                    op.attributes["op_name"] = StringAttr.get(f"kernel_{node_id}")

                del op.attributes["top_level"]
                del op.attributes["node_id"]

            for nested_op in op.body.operations:
                process_op(nested_op, func_name)
        elif hasattr(op, "regions"):
            for region in op.regions:
                for block in region.blocks:
                    for nested_op in block.operations:
                        process_op(nested_op, func_name)

    with module.context:
        for func in module.body.operations:
            if not isinstance(func, func_d.FuncOp):
                continue

            func_name = func.name.value

            for op in func.body.blocks[0]:
                process_op(op, func_name)

    return module, node_to_fn


def extract_reorder_and_pipeline(
    analysis_result: DFGAnalysisResult,
    dfg: DFG,
    node_to_fn: dict[int, str],
    schedule: Schedule,
) -> list[SchedulePrimitive]:
    schedule_primitives = []
    permutations = analysis_result.loop_permutations

    for node_id, perm_idx in permutations:
        loop_band_collection = list(
            v for _, v in schedule.get_loops(node_to_fn[node_id])
        )
        # support only perfect affine kernels
        assert (
            len(loop_band_collection) == 1
        ), "Only perfect affine kernels are supported"
        loop_band = list(loop_band_collection[0].loops.values())

        node_info: NodeInfo = dfg.get_node(node_id).node_info[perm_idx]
        if perm_idx == 0:
            schedule_primitives.append(SchedulePrimitive.pipeline(loop_band[-1], 1))
        else:
            perm = node_info.permutation
            new_loop_order = [loop_band[i] for i in perm]
            schedule_primitives.append(SchedulePrimitive.reorder(new_loop_order))
            schedule_primitives.append(
                SchedulePrimitive.pipeline(new_loop_order[-1], 1)
            )

    return schedule_primitives


def extract_buffer_to_fifo(
    analysis_result: DFGAnalysisResult, dfg: DFG, schedule: Schedule, top_func: str
) -> list[SchedulePrimitive | UnresolvedFIFOPrimitive]:
    permutations = dict(analysis_result.loop_permutations)
    result = []
    for node_idx, node in dfg.nodes.items():
        if node.type != DFGNodeType.AFFINE:
            continue
        dst_node_info = node.node_info[permutations[node_idx]]
        for edge in dfg.in_edges[node_idx]:
            src_node = dfg.nodes[edge.id]
            if src_node.type != DFGNodeType.AFFINE:
                continue
            src_node_info = src_node.node_info[permutations[edge.id]]
            memref = edge.value
            assert memref in dst_node_info.loads_map
            assert memref in src_node_info.stores_map
            if (
                dst_node_info.loads_map[memref].access_map
                == src_node_info.stores_map[memref].access_map
            ) and not isinstance(memref.owner, Block):
                assert (
                    "name" in memref.owner.attributes
                ), f"Buffer {memref.owner} has no name"

                # Check tiling factor to see if buffer needs to be converted to
                # array of FIFOs
                buffer_name = memref.owner.attributes["name"].value

                if analysis_result.tiling_factors is not None:
                    node_tiling_factors = analysis_result.tiling_factors.get(
                        node_idx, []
                    )
                    new_buffer, n_dims = create_fifo_array(
                        schedule,
                        memref.owner,
                        node_tiling_factors,
                        dst_node_info.loads_map[memref].op,
                        node.loop_info,
                        buffer_name,
                    )
                    if n_dims < 0:
                        # If n_dims is negative, it means the buffer is not tiled
                        _insert_guard(
                            src_node_info.stores_map[memref].op, src_node.loop_info
                        )
                        _insert_guard(
                            dst_node_info.loads_map[memref].op, node.loop_info
                        )
                        result.append(UnresolvedFIFOPrimitive(buffer_name, node_idx))
                        continue

                    new_buffer_uses = list(new_buffer.result.uses)
                    new_load = [
                        use
                        for use in new_buffer_uses
                        if isinstance(use.owner, affine_d.AffineLoadOp)
                    ]
                    new_store = [
                        use
                        for use in new_buffer_uses
                        if isinstance(use.owner, affine_d.AffineStoreOp)
                    ]
                    assert (
                        len(new_load) == 1
                    ), f"Expected one load for {buffer_name}, found {len(new_load)}"
                    assert (
                        len(new_store) == 1
                    ), f"Expected one store for {buffer_name}, found {len(new_store)}"

                    _insert_guard(new_store[0].owner, src_node.loop_info)
                    _insert_guard(new_load[0].owner, node.loop_info)

                    result.append(
                        SchedulePrimitive.buffer_to_fifo(
                            MockBuffer(top_func, buffer_name),
                            list(range(n_dims // 2)),
                            0,
                        )
                    )

                else:
                    _insert_guard(
                        src_node_info.stores_map[memref].op, src_node.loop_info
                    )
                    _insert_guard(dst_node_info.loads_map[memref].op, node.loop_info)

                    result.append(UnresolvedFIFOPrimitive(buffer_name, node_idx))
            else:
                # TODO: probably can use a partition instead of a fifo here based on the
                # loop tiling here?
                continue

    return result


def _get_affine_if_op(op):
    parent = op.parent
    while parent is not None and not isinstance(parent.opview, affine_d.AffineIfOp):
        parent = parent.parent
    return parent


def _insert_guard(op: Operation, loops: list[LoopInfo]):
    op = op.opview
    assert isinstance(op, (affine_d.AffineLoadOp, affine_d.AffineStoreOp))
    affine_if_op = _get_affine_if_op(op)
    if not affine_if_op:
        # insert guard before the fifo write/read
        irrelevant_loops = [
            loop
            for loop in loops
            if loop.op.opview.induction_variable not in op.indices
        ]
        if not irrelevant_loops:
            return
        with op.context:
            if isinstance(op, affine_d.AffineLoadOp):
                guard_condition = affine_d.IntegerSet.get(
                    len(irrelevant_loops),
                    0,
                    [
                        affine_d.AffineExpr.get_dim(i) - loop_info.lower_bound
                        for i, loop_info in enumerate(irrelevant_loops)
                    ],
                    [True] * len(irrelevant_loops),
                )
            else:
                guard_condition = affine_d.IntegerSet.get(
                    len(irrelevant_loops),
                    0,
                    [
                        affine_d.AffineExpr.get_dim(i)
                        - loop_info.upper_bound
                        + loop_info.step
                        for i, loop_info in enumerate(irrelevant_loops)
                    ],
                    [True] * len(irrelevant_loops),
                )

            with InsertionPoint(op) as ip:
                guard_if = affine_d.AffineIfOp(
                    results_=[],
                    _gen_arg_0=[
                        loop.op.opview.induction_variable for loop in irrelevant_loops
                    ],
                    loc=Location.unknown(),
                    ip=ip,
                )

                guard_if.attributes["condition"] = IntegerSetAttr.get(guard_condition)

                then_block = Block.create_at_start(parent=guard_if.thenRegion)
                yield_op = affine_d.AffineYieldOp(
                    [],
                    ip=InsertionPoint.at_block_begin(then_block),
                    loc=Location.unknown(),
                )
                op.move_before(yield_op)
            if isinstance(op, affine_d.AffineLoadOp):
                # Find the enclosing func via Block -> Region -> owner op
                blk = op.parent  # this is a Block
                region = getattr(blk, "parent", None)
                owner_op = getattr(region, "owner", None)
                assert owner_op is not None and hasattr(owner_op, "opview"), "Failed to find enclosing function op"
                assert isinstance(owner_op.opview, func_d.FuncOp), "Enclosing op is not a func.func"
                fn_op = owner_op  # operation whose opview is func.func

                alloc_op = memref_d.AllocOp(
                    op.memref.type,
                    [],
                    [],
                    ip=InsertionPoint.at_block_begin(fn_op.opview.body.blocks[0]),
                    loc=Location.unknown(),
                )

                # memref load and store to this alloc in the if statement
                with InsertionPoint.at_block_terminator(then_block) as ip:
                    store_op = affine_d.AffineStoreOp(
                        op.result,
                        alloc_op.result,
                        op.indices,
                        op.map,
                        ip=ip,
                        loc=Location.unknown(),
                    )
                    load_op = affine_d.AffineLoadOp(
                        op.result.type,
                        alloc_op.result,
                        op.indices,
                        op.map,
                        ip=ip,
                        loc=Location.unknown(),
                    )
                load_op.move_after(guard_if)
                for use in op.result.uses:
                    if use.owner.operation != store_op:
                        use.owner.operation.replace_uses_of_with(
                            op.result, load_op.result
                        )

def _is_block_argument(val) -> bool:
    """
    Robustly detect MLIR block arguments across builds.
    Works whether or not BlockArgument is a distinct Python class.
    """
    # Fast path: MLIR Value typically exposes this flag
    try:
        iba = getattr(val, "is_block_argument", None)
        if iba is True:
            return True
    except Exception:
        pass

    # If the owner is a Block, this is a BlockArgument (OpResult owners are Operations)
    owner = getattr(val, "owner", None)
    try:
        from .._mlir.ir import Block as _Block
        if isinstance(owner, _Block):
            return True
    except Exception:
        # Fallback: structural check
        if owner is not None and hasattr(owner, "arguments") and hasattr(owner, "operations"):
            # Typical Block shape in Python bindings
            if hasattr(val, "arg_number"):
                return True

    return False


def _map_blockarg_to_top_name_and_func(module, schedule, top_func_name, callee_func_name, callee_arg_index):
    """
    Given a callee function and one of its BlockArguments (by index),
    find the callsite in the top function and return the *top-level* buffer identity:
      -> (top_func_name, top_arg_name_or_index_str)
    If the actual operand is a memref.cast chain, peel it.
    Returns (func_name, buf_name) or (None, None) if unresolved.
    """
    # find top function op
    topf = None
    for f in module.body.operations:
        if isinstance(f, func_d.FuncOp) and f.name.value == top_func_name:
            topf = f
            break
    if topf is None:
        return (None, None)

    # find the callee func op
    callee = None
    for f in module.body.operations:
        if isinstance(f, func_d.FuncOp) and f.name.value == callee_func_name:
            callee = f
            break
    if callee is None:
        return (None, None)

    # look for callsites in the top function
    for op in topf.body.blocks[0].operations:
        if not isinstance(op, func_d.CallOp):
            continue
        if op.attributes.get("callee", None) is None:
            continue
        if op.attributes["callee"].value != callee_func_name:
            continue
        if callee_arg_index >= len(op.operands):
            continue
        actual = op.operands[callee_arg_index]

        # peel memref.cast/subview/reinterpret_cast to a stable root
        root = actual
        seen = set()
        while hasattr(root, "owner") and root.owner not in seen:
            seen.add(root.owner)
            o = root.owner
            nm = getattr(o, "name", getattr(o.operation, "name", ""))
            if nm in ("memref.cast", "memref.reinterpret_cast", "memref.subview"):
                root = o.operands[0]
                continue
            break

        # top-level BlockArgument?
        if _is_block_argument(root):
            top_idx = root.arg_number
            fa = schedule.func_args.get(top_func_name, [])
            if top_idx < len(fa):
                friendly = getattr(fa[top_idx], "name", fa[top_idx]) or str(top_idx)
            else:
                friendly = str(top_idx)
            return (top_func_name, friendly)

        # alloc/global with name?
        if hasattr(root, "owner") and hasattr(root.owner, "attributes") and "name" in root.owner.attributes:
            return (top_func_name, root.owner.attributes["name"].value)

    return (None, None)


def extract_array_partitions(
    analysis_result: DFGAnalysisResult,
    dfg: DFG,
    top_func_name: str,
    schedule: Schedule,
) -> list[SchedulePrimitive]:
    def _resolve_source_memref(val):
        """Follow memref.cast/subview/etc. back to a stable identity."""
        if _is_block_argument(val):
            return val
        op = getattr(val, "owner", None)
        while op is not None and not isinstance(op, Block):
            op_name = op.operation.name if hasattr(op, "operation") else getattr(op, "name", "")
            if op_name in ("memref.cast", "memref.reinterpret_cast", "memref.subview"):
                src = op.operands[0]
                if _is_block_argument(src):
                    return src
                val = src
                op = getattr(val, "owner", None)
                continue
            break
        return val

    tiling = analysis_result.tiling_factors
    if not tiling:
        print("[array_partitions] No tiling factors found")
        return []

    print(f"[array_partitions] Tiling nodes: {list(tiling.keys())}")
    permutations = dict(analysis_result.loop_permutations)
    per_buf_dim_factor: dict[tuple[str, int], int] = {}
    owner_for_buf: dict[str, str] = {}

    for node_id, node in dfg.nodes.items():
        if node.type != DFGNodeType.AFFINE:
            continue
        if node_id not in tiling:
            continue

        tf = {depth: factor for depth, factor in tiling[node_id] if factor > 1}
        if not tf:
            continue

        print(f"[array_partitions] Node {node_id} tiling: {tf}")

        perm_idx = permutations.get(node_id, 0)
        node_info: NodeInfo = node.node_info[perm_idx]

        for access_map in (node_info.loads_map, node_info.stores_map):
            for access_map_name, acc in access_map.items():
                opv = acc.op.opview
                if isinstance(opv, affine_d.AffineLoadOp):
                    memref_val = opv.memref
                elif isinstance(opv, affine_d.AffineStoreOp):
                    memref_val = opv.memref
                else:
                    continue

                root = _resolve_source_memref(memref_val)
                print(f"  [debug] Node {node_id} {access_map_name}: root={root}")

                buffer_name = None
                buffer_func = None
                arg_index = None

                # =========== BLOCK ARGUMENT (function parameter) ===========
                if _is_block_argument(root):
                    parent_block = root.owner  # Block
                    parent_region = getattr(parent_block, "parent", None)  # Region
                    parent_owner_op = getattr(parent_region, "owner", None)  # Operation
                    if parent_owner_op is not None and hasattr(parent_owner_op, "opview") and isinstance(parent_owner_op.opview, func_d.FuncOp):
                        callee_func_name = parent_owner_op.opview.name.value
                    else:
                        callee_func_name = top_func_name
                    try:
                        arg_index = getattr(root, "arg_number", None)
                        if arg_index is None and hasattr(root, "owner") and hasattr(root.owner, "arguments"):
                            # fallback: manually locate argument index in the block
                            block_args = list(root.owner.arguments)
                            arg_index = block_args.index(root)
                    except Exception:
                        arg_index = None
                    print(f"  [debug] root is BlockArgument #{arg_index} of func {callee_func_name}")

                    # top-level argument (A, B, x, y) handling
                    if callee_func_name == top_func_name:
                        fa = schedule.func_args.get(top_func_name, [])
                        print(f"  [debug] top-level func args for {top_func_name}: {fa}")
                        if arg_index < len(fa):
                            buffer_func = top_func_name
                            arg_obj = fa[arg_index]
                            # ✅ Handle DTensor or plain string
                            if hasattr(arg_obj, "name"):
                                buffer_name = arg_obj.name
                            else:
                                buffer_name = str(arg_obj)
                            print(f"  [debug] matched top-level arg {buffer_name} for index {arg_index}")
                        else:
                            buffer_func = top_func_name
                            buffer_name = f"arg{arg_index}"
                            print(f"  [debug] fallback arg name arg{arg_index}")
                    else:
                        # nested call argument mapping
                        print(f"  [debug] mapping callee {callee_func_name} arg {arg_index} to top...")
                        mapped_func, mapped_name = _map_blockarg_to_top_name_and_func(
                            schedule.module, schedule, top_func_name, callee_func_name, arg_index
                        )
                        if mapped_func and mapped_name:
                            buffer_func = mapped_func
                            buffer_name = mapped_name
                            print(f"  [debug] resolved via map → {mapped_func}:{mapped_name}")
                        else:
                            # fallback local naming
                            buffer_func = callee_func_name
                            fa = schedule.func_args.get(buffer_func, [])
                            if arg_index < len(fa):
                                buffer_name = getattr(fa[arg_index], "name", fa[arg_index]) or str(arg_index)
                            else:
                                buffer_name = str(arg_index)
                            print(f"  [debug] fallback local param name {buffer_name} for {callee_func_name}")
                # =========== LOCAL ALLOC OR GLOBAL ===========
                else:
                    if hasattr(root, "owner") and hasattr(root.owner, "attributes") and "name" in root.owner.attributes:
                        cur = root.owner
                        owner_func = None
                        while cur is not None:
                            po = getattr(cur, "parent", None)
                            if po is None:
                                break
                            if hasattr(po, "opview") and isinstance(po.opview, func_d.FuncOp):
                                owner_func = po.opview.name.value
                                break
                            cur = po
                        buffer_func = owner_func or top_func_name
                        buffer_name = root.owner.attributes["name"].value
                        print(f"  [debug] local alloc {buffer_name} under {buffer_func}")
                    else:
                        print(f"  [debug] could not identify alloc root {root}")

                # Record owner only if both valid
                if buffer_name and buffer_func:
                    owner_for_buf.setdefault(buffer_name, buffer_func)
                else:
                    print(f"  [skip] missing owner or name: func={buffer_func}, name={buffer_name}")

                if not buffer_name or buffer_func is None:
                    print(f"  [skip] could not derive name for {root}")
                    continue

                # extra top-level debug
                print(f"  [debug] derived buffer_func={buffer_func}, buffer_name={buffer_name}")

                print(f"  [access] Node {node_id} {access_map_name} → {buffer_name}")

                # Record per-dimension partition factors
                indices = list(opv.indices)
                for depth, loop_info in enumerate(node.loop_info):
                    iv = loop_info.op.opview.induction_variable
                    if depth in tf and iv in indices:
                        dim = depth + 1
                        key = (buffer_name, dim)
                        per_buf_dim_factor[key] = max(per_buf_dim_factor.get(key, 1), tf[depth])
                        print(f"    [match] buffer={buffer_name}, dim={dim}, factor={tf[depth]}")

    print(f"[array_partitions] Final partitions: {per_buf_dim_factor}")
    print(f"[array_partitions] owner_for_buf mapping: {owner_for_buf}")

    # --- Infer ranks from schedule.func_args to avoid flattening 2D buffers ---
    inferred_ranks: dict[str, int] = {}
    top_args = schedule.func_args.get(top_func_name, [])
    for arg in top_args:
        name = getattr(arg, "name", None)
        if name is not None and hasattr(arg, "shape"):
            inferred_ranks[name] = len(arg.shape)
    print(f"  [rank-infer] inferred ranks: {inferred_ranks}")

    # --- Normalize only when necessary ---
    # Fixes invalid dims for 1D buffers (like x, y) but preserves multi-dim buffers (like A, B)
    normalized: dict[tuple[str, int], int] = {}
    for (buf, dim), factor in per_buf_dim_factor.items():
        rank = inferred_ranks.get(buf, 1)
        if dim > rank:
            valid_dim = rank
            print(f"  [normalize] adjusted dim for {buf}: {dim} → {valid_dim} (rank={rank})")
        else:
            valid_dim = dim

        key = (buf, valid_dim)
        normalized[key] = max(normalized.get(key, 1), factor)

    per_buf_dim_factor = normalized
    # --- End normalization ---

    # Group by buffer
    by_buf: dict[str, dict[int, int]] = {}
    for (buf, dim), factor in per_buf_dim_factor.items():
        by_buf.setdefault(buf, {})[dim] = factor

    prims: list[SchedulePrimitive] = []
    for buf, dim2factor in sorted(by_buf.items()):
        dims = sorted(dim2factor.keys())
        factors = {dim2factor[d] for d in dims}
        print(f"  [debug] buffer {buf}: dims={dims}, factors={factors}, owner={owner_for_buf.get(buf)}")
        if len(dims) >= 2 and len(factors) == 1 and next(iter(factors)) > 1:
            factor = next(iter(factors))
            prims.append(
                SchedulePrimitive.partition(
                    target=MockBuffer(owner_for_buf.get(buf, top_func_name), buf),
                    partition_type=Partition.Cyclic,
                    dim=0,
                    factor=factor,
                )
            )
        else:
            for dim in dims:
                factor = dim2factor[dim]
                if factor > 1:
                    prims.append(
                        SchedulePrimitive.partition(
                            target=MockBuffer(owner_for_buf.get(buf, top_func_name), buf),
                            partition_type=Partition.Cyclic,
                            dim=dim,
                            factor=factor,
                        )
                    )

    print(f"[array_partitions] Created {len(prims)} partition primitives.")
    return prims

# Node-parallel specific code
def extract_tiling(
    analysis_results: DFGAnalysisResult,
    node_to_fn: dict[int, str],
    schedule: Schedule,
):
    tiling_factors = analysis_results.tiling_factors

    if not tiling_factors:
        return []

    tiling_primitives = []
    for node_id, factors in tiling_factors.items():
        fn_name = node_to_fn[node_id]
        loop_band_collection = list(v for _, v in schedule.get_loops(fn_name))

        assert len(loop_band_collection) == 1, "Only perfect affine kernels supported"

        loop_band = list(loop_band_collection[0].loops.values())

        for depth, factor in sorted(factors):
            if factor > 1:
                loop = loop_band[depth]
                tiling_primitives.append(SchedulePrimitive.split(loop, factor))

    return tiling_primitives


def create_fifo_array(
    schedule: Schedule,
    old_alloc: Operation,
    tiling_factors: list[tuple[int, int]],
    dst_op: Operation,
    dst_loop_info: list[LoopInfo],
    buffer_name: str,
) -> tuple[memref_d.AllocOp, int]:
    """Create a FIFO array to replace the original buffer."""
    tiling_factors = dict(tiling_factors)
    relevant_loop_depths = [
        i
        for i, loop in enumerate(dst_loop_info)
        if loop.op.opview.induction_variable in dst_op.opview.indices
    ]
    fifo_dims = [tiling_factors[depth] for depth in relevant_loop_depths]
    original_dims = old_alloc.result.type.shape
    extra_dims = [
        original_dim // fifo_dim
        for original_dim, fifo_dim in zip(original_dims, fifo_dims)
    ]

    if all(dim == 1 for dim in fifo_dims):
        return old_alloc, -1

    with schedule.module.context, Location.unknown():
        old_type = old_alloc.result.type
        element_type = old_type.element_type

        fifo_type = MemRefType.get(fifo_dims + extra_dims, element_type)

        # Create new allocation at the same location as the old one
        ip = InsertionPoint(old_alloc)
        new_alloc = memref_d.AllocOp(fifo_type, [], [], ip=ip)
        new_alloc.attributes["name"] = StringAttr.get(buffer_name)

        # replace old allocation with new FIFO array in the schedule
        uses_to_update = list(old_alloc.result.uses)
        for use in uses_to_update:
            op = use.owner.opview

            if isinstance(op, (affine_d.AffineLoadOp, affine_d.AffineStoreOp)):
                indices = list(op.indices)

                fifo_exprs = []
                extra_exprs = []
                for i, dim_size in enumerate(fifo_dims):
                    if dim_size > 1:
                        fifo_exprs.append(
                            affine_d.AffineExpr.get_mod(
                                affine_d.AffineExpr.get_dim(i),
                                affine_d.AffineExpr.get_constant(dim_size),
                            )
                        )
                    else:
                        fifo_exprs.append(affine_d.AffineExpr.get_dim(i))

                for i, dim_size in enumerate(fifo_dims):
                    if dim_size > 1:
                        extra_exprs.append(
                            affine_d.AffineExpr.get_floor_div(
                                affine_d.AffineExpr.get_dim(i),
                                affine_d.AffineExpr.get_constant(dim_size),
                            )
                        )
                    else:
                        extra_exprs.append(affine_d.AffineExpr.get_dim(i))

                mod_map = affine_d.AffineMap.get(
                    len(fifo_exprs), 0, fifo_exprs + extra_exprs
                )

                if isinstance(op, affine_d.AffineLoadOp):
                    new_op = affine_d.AffineLoadOp(
                        op.result.type,
                        new_alloc.result,
                        indices,
                        map=mod_map,
                        ip=InsertionPoint(op),
                    )
                    op.result.replace_all_uses_with(new_op.result)

                else:  # AffineStoreOp
                    new_op = affine_d.AffineStoreOp(
                        op.value,
                        new_alloc.result,
                        indices,
                        map=mod_map,
                        ip=InsertionPoint(op),
                    )

                for attr_name in ("from", "to"):
                    if attr_name in op.attributes:
                        new_op.attributes[attr_name] = op.attributes[attr_name]

                op.erase()
        old_alloc.erase()

    return new_alloc, len(fifo_dims) + len(
        extra_dims
    )  # return the number of dimensions in the new FIFO array


def _fix_call_memref_layout_mismatches(module: Module):
    """
    After array partitioning (which changes affine maps/layouts on caller-side memrefs),
    insert memref.cast at callsites so the argument type matches the callee param type.
    Only fixes *layout* differences (shape/elemtype/memspace must match).
    """
    with module.context, Location.unknown():
        for f in module.body.operations:
            if not isinstance(f, func_d.FuncOp):
                continue
            # Walk call ops in this function
            for op in list(f.entry_block.operations):
                if not isinstance(op, func_d.CallOp):
                    continue
                callee_sym = op.attributes.get("callee", None)
                if callee_sym is None:
                    continue
                callee_name = callee_sym.value
                callee = None
                for g in module.body.operations:
                    if isinstance(g, func_d.FuncOp) and g.name.value == callee_name:
                        callee = g
                        break
                if callee is None:
                    continue
                changed = False
                new_ops = list(op.operands)
                for i, (actual, formal) in enumerate(zip(op.operands, callee.arguments)):
                    src_ty = getattr(actual, "type", None)
                    dst_ty = getattr(formal, "type", None)
                    if isinstance(src_ty, MemRefType) and isinstance(dst_ty, MemRefType):
                        same_shape = tuple(src_ty.shape) == tuple(dst_ty.shape)
                        same_elem  = src_ty.element_type == dst_ty.element_type
                        same_space = src_ty.memory_space == dst_ty.memory_space
                        if same_shape and same_elem and same_space and src_ty != dst_ty:
                            with InsertionPoint(op):
                                cast = memref_d.CastOp(dst_ty, actual)
                            new_ops[i] = cast.result
                            changed = True
                if changed:
                    op.operands[:] = new_ops


def _tiling_post_process(schedule: Schedule):
    """
    Post-process the schedule after tiling to ensure all inner loops are placed after all outer loops and are fully unrolled.
    """
    primitives = []

    for func in schedule.module.body.operations:
        if not isinstance(func, func_d.FuncOp):
            continue

        func_name = func.name.value

        loop_bands = list(v for _, v in schedule.get_loops(func_name))
        if len(loop_bands) == 0:
            continue
        assert len(loop_bands) == 1, "Only perfect affine kernels supported"

        loop_band = loop_bands[0]
        loops = list(loop_band.loops.values())

        outer_loops = []
        inner_loops = []
        outer_indices = []
        inner_indices = []

        for i, loop_wrapper in enumerate(loops):
            loop = loop_wrapper.loop
            if "loop_name" in loop.attributes:
                loop_name = loop.attributes["loop_name"].value
                if loop_name.endswith(".outer"):
                    outer_loops.append(loop_wrapper)
                    outer_indices.append(i)
                elif loop_name.endswith(".inner"):
                    inner_loops.append(loop_wrapper)
                    inner_indices.append(i)

        # Skip if no split loops found
        if not outer_loops or not inner_loops:
            continue

        # Check if reordering is needed
        if outer_indices and inner_indices:
            if min(inner_indices) < max(outer_indices):
                # Create new loop order: all outer loops first, then all inner loops
                new_loop_order = []

                for i, loop in enumerate(loops):
                    if i in outer_indices:
                        new_loop_order.append(loop)

                for i, loop in enumerate(loops):
                    if i in inner_indices:
                        new_loop_order.append(loop)

                for i, loop in enumerate(loops):
                    if i not in outer_indices and i not in inner_indices:
                        new_loop_order.append(loop)

                primitives.append(SchedulePrimitive.reorder(new_loop_order))

                loops = new_loop_order

        for loop in inner_loops:
            primitives.append(SchedulePrimitive.unroll(loop, 0))

        if outer_loops:
            innermost_outer = outer_loops[-1]
            primitives.append(SchedulePrimitive.pipeline(innermost_outer, 1))

    return primitives
