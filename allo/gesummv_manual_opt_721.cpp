//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

void affine_kernel(
  float v0,
  hls::stream< float > v1[10] /* v1[10][25] */
) { // L7
  #pragma HLS stream variable=v1 depth=-1
  l_init_A_i_outer: for (int i_outer = 0; i_outer < 25; i_outer++) { // was 50
  #pragma HLS pipeline II=1
    l_i_inner: for (int i_inner = 0; i_inner < 10; i_inner++) { // was 5
    #pragma HLS unroll
      v1[i_inner].write(v0); // v1[i_inner][i_outer] = v0;
    }
  }
}

void affine_kernel_0(
  hls::stream< float > v4[10] /* v4[10][25] */,
  float v5[250],
  float v6[250][250],
  float v7[250],
  hls::stream< float > v8[10] /* v8[10][25] */
) { // (was L15-ish)
  #pragma HLS stream variable=v4 depth=-1
  #pragma HLS stream variable=v8 depth=-1

  // accumulator vector
  #pragma HLS array_partition variable=v5 cyclic dim=1 factor=10

  // matrix and x
  #pragma HLS array_partition variable=v6 cyclic dim=1 factor=10
  #pragma HLS array_partition variable=v6 cyclic dim=2 factor=10
  #pragma HLS array_partition variable=v7 cyclic dim=1 factor=10

  // j tile
  l_Ax_j_outer: for (int j_outer = 0; j_outer < 25; j_outer++) {
    // i tile
    l_i_outer1: for (int i_outer1 = 0; i_outer1 < 25; i_outer1++) {
      #pragma HLS pipeline II=1

      // unroll i lanes (10-wide)
      l_i_inner1: for (int i_inner1 = 0; i_inner1 < 10; i_inner1++) {
        #pragma HLS unroll

        const int i = i_inner1 + (i_outer1 * 10);

        // Initialize accumulator exactly once per i (when starting j_outer==0),
        // otherwise load prior partial sum from v5.
        float acc;
        if (j_outer == 0) {
          acc = v4[i_inner1].read();   // produced by your init kernel (usually 0.0)
        } else {
          acc = v5[i];
        }

        // Compute this j-tile's contribution with an unrolled reduction.
        float sum = 0.0f;
        l_j_inner: for (int j_inner = 0; j_inner < 10; j_inner++) {
          #pragma HLS unroll
          const int j = j_inner + (j_outer * 10);
          sum += v6[i][j] * v7[j];
        }

        acc += sum;
        v5[i] = acc;

        // Emit output only after the final j tile
        if (j_outer == 24) {
          v8[i_inner1].write(acc);
        }
      }
    }
  }
}

void affine_kernel_1(
  float v20,
  hls::stream< float > v21[10] /* v21[10][25] */
) { // L40
  #pragma HLS stream variable=v21 depth=-1
  l_init_B_i_outer2: for (int i_outer2 = 0; i_outer2 < 25; i_outer2++) { // was 50
  #pragma HLS pipeline II=1
    l_i_inner2: for (int i_inner2 = 0; i_inner2 < 10; i_inner2++) { // was 5
    #pragma HLS unroll
      v21[i_inner2].write(v20);
    }
  }
}

void affine_kernel_2(
  hls::stream< float > v24[10] /* v24[10][25] */,
  float v25[250],
  float v26[250][250],
  float v27[250],
  hls::stream< float > v28[10] /* v28[10][25] */
) {
  #pragma HLS stream variable=v24 depth=-1
  #pragma HLS stream variable=v28 depth=-1

  // accumulator vector
  #pragma HLS array_partition variable=v25 cyclic dim=1 factor=10

  // matrix and x
  #pragma HLS array_partition variable=v26 cyclic dim=1 factor=10
  #pragma HLS array_partition variable=v26 cyclic dim=2 factor=10
  #pragma HLS array_partition variable=v27 cyclic dim=1 factor=10

  l_Bx_j_outer1: for (int j_outer1 = 0; j_outer1 < 25; j_outer1++) {
    l_i_outer3: for (int i_outer3 = 0; i_outer3 < 25; i_outer3++) {
      #pragma HLS pipeline II=1

      l_i_inner3: for (int i_inner3 = 0; i_inner3 < 10; i_inner3++) {
        #pragma HLS unroll

        const int i = i_inner3 + (i_outer3 * 10);

        // init accumulator once per i (first j tile), else load partial sum
        float acc;
        if (j_outer1 == 0) {
          acc = v24[i_inner3].read();   // typically 0.0 from init kernel
        } else {
          acc = v25[i];
        }

        // unrolled reduction across this j tile
        float sum = 0.0f;
        l_j_inner1: for (int j_inner1 = 0; j_inner1 < 10; j_inner1++) {
          #pragma HLS unroll
          const int j = j_inner1 + (j_outer1 * 10);
          sum += v26[i][j] * v27[j];
        }

        acc += sum;
        v25[i] = acc;

        // emit output only after last j tile
        if (j_outer1 == 24) {
          v28[i_inner3].write(acc);
        }
      }
    }
  }
}

void affine_kernel_3(
  hls::stream< float > v40[10] /* v40[10][25] */,
  hls::stream< float > v41[10] /* v41[10][25] */,
  float v42[250]
) { // L73
  #pragma HLS stream variable=v40 depth=-1
  #pragma HLS stream variable=v41 depth=-1
  #pragma HLS array_partition variable=v42 cyclic dim=1 factor=10  // was 5

  l_axpby_i_outer4: for (int i_outer4 = 0; i_outer4 < 25; i_outer4++) { // was 50
  #pragma HLS pipeline II=1
    l_i_inner4: for (int i_inner4 = 0; i_inner4 < 10; i_inner4++) { // was 5
    #pragma HLS unroll
      float v45 = v40[i_inner4].read();
      float v46 = v41[i_inner4].read();
      float v47 = v45 + v46;
      v42[(i_inner4 + (i_outer4 * 10))] = v47; // was *5
    }
  }
}

void kernel_gesummv(
  float v48[250][250],
  float v49[250][250],
  float v50[250],
  float v51[250]
) { // L84
  #pragma HLS dataflow
  #pragma HLS array_partition variable=v48 cyclic dim=1 factor=10  // was 5
  #pragma HLS array_partition variable=v48 cyclic dim=2 factor=10

  #pragma HLS array_partition variable=v49 cyclic dim=1 factor=10  // was 5
  #pragma HLS array_partition variable=v49 cyclic dim=2 factor=10

  #pragma HLS array_partition variable=v50 cyclic dim=1 factor=10

  #pragma HLS array_partition variable=v51 cyclic dim=1 factor=10  // was 5

  float _buffer_0[250];
  #pragma HLS array_partition variable=_buffer_0 cyclic dim=1 factor=10 // was 5

  hls::stream< float > _buffer_1[10] /* _buffer_1[10][25] */;
  #pragma HLS stream variable=_buffer_1 depth=-1

  hls::stream< float > tmp[10] /* tmp[10][25] */;
  #pragma HLS stream variable=tmp depth=-1

  float _buffer_2[250];
  #pragma HLS array_partition variable=_buffer_2 cyclic dim=1 factor=10 // was 5

  hls::stream< float > _buffer_3[10] /* _buffer_3[10][25] */;
  #pragma HLS stream variable=_buffer_3 depth=-1

  hls::stream< float > yB[10] /* yB[10][25] */;
  #pragma HLS stream variable=yB depth=-1

  affine_kernel((float)0.000000, tmp);
  affine_kernel_0(tmp, _buffer_0, v48, v50, _buffer_1);
  affine_kernel_1((float)0.000000, yB);
  affine_kernel_2(yB, _buffer_2, v49, v50, _buffer_3);
  affine_kernel_3(_buffer_1, _buffer_3, v51);
}
