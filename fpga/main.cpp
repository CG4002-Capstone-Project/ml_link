#include "ap_fixed.h"
#include <algorithm>

// #define DEBUG

#define N_DANCERS 1

#define FC1_IN 84
#define FC1_OUT 64
#define FC2_OUT 16
#define FC3_OUT 3

#include "types.h"
// #include "buffer.h"
// #include "conv.h"
// #include "relu.h"
#include "linear.h"
// #include "maxpool.h"

void technoedge(
	d_t (&sensor_data)[N_DANCERS][84],
	weights_t &wts,
	dbg_t &dbg,
	result_t &result
) {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=sensor_data
#pragma HLS INTERFACE s_axilite port=wts
#pragma HLS INTERFACE s_axilite port=dbg
#pragma HLS INTERFACE s_axilite port=result

	// FC1
	d_t fc1_output[N_DANCERS][FC1_OUT];
	linear<N_DANCERS, FC1_IN, FC1_OUT>(
		sensor_data,
		fc1_output,
		wts.fc1_wt,
		wts.fc1_bias
	);

	// FC2
	d_t fc2_output[N_DANCERS][FC2_OUT];
	linear<N_DANCERS, FC1_OUT, FC2_OUT>(
		fc1_output,
		fc2_output,
		wts.fc2_wt,
		wts.fc2_bias
	);

	// FC3
	linear<N_DANCERS, FC2_OUT, FC3_OUT>(
		fc2_output,
		result.result,
		wts.fc3_wt,
		wts.fc3_bias
	);
}
