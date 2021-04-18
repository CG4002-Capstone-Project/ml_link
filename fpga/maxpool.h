#include "types.h"
#include <algorithm>

template <int dim1, int dim2, int dim3>
void maxpool(d_t (&input)[dim1][dim2][dim3], d_t (&output)[dim1][dim2 >> 1][dim3]) {
	for (int i = 0; i < dim1; i++) {
		for (int k = 0; k < dim3; k++) {
			max_op: for (int j = 0; j < dim2/2; j++) {
#pragma HLS PIPELINE
				output[i][j][k] = std::max(input[i][2*j][k], input[i][2*j+1][k]);
			}
		}
	}
}

