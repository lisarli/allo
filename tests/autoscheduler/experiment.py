from allo.autoscheduler.util import check_single_producer_single_consumer
from gurobipy import GurobiError
import numpy as np
import pytest
from allo.ir.types import float32, int32
from allo.autoscheduler.passes import dataflow_optimization_pass, DEBUG_POINTS
from allo.autoscheduler.config import AutoschedulerConfig
from tests.autoscheduler.polybench import get_polybench
import allo
from allo.backend.hls import is_available


if __name__ == "__main__":
    # debug_point = None
    # schedule, inputs, expected = get_polybench(
    #     "atax", size="medium", concrete_type=float32
    # )
    # cfg = (AutoschedulerConfig.builder().with_debug_point(debug_point).with_kind("graph").with_verbose(True))
    # optimized_schedule = dataflow_optimization_pass(schedule, cfg)
    # print("--- module after optimized schedule: ---")
    # print(optimized_schedule.module)
    # print("--- end module after optimized schedule ---")
    # # assert check_single_producer_single_consumer(optimized_schedule.module), "module after optimized schedule failed spsc check"
    # mod = optimized_schedule.build(
    #     target="vhls", mode="sw_emu", project="test_atax.prj", wrap_io=True
    # )

    # A, x = inputs
    # y = np.zeros_like(expected)
    # mod(A, x, y)
    # np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    debug_point = None
    schedule, inputs, expected = get_polybench(
        "gesummv", size="medium", concrete_type=float32
    )
    cfg = (AutoschedulerConfig.builder().with_debug_point(debug_point).with_kind("combined").with_verbose(True))
    optimized_schedule = dataflow_optimization_pass(schedule, cfg)
    print("--- module after optimized schedule: ---")
    print(optimized_schedule.module)
    print("--- end module after optimized schedule ---")
    # assert check_single_producer_single_consumer(optimized_schedule.module), "module after optimized schedule failed spsc check"
    mod = optimized_schedule.build(
        target="vhls", mode="sw_emu", project="test_gesummv.prj", wrap_io=True
    )

    A, B, x = inputs
    y = np.zeros_like(expected)
    mod(A, B, x, y)
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)
