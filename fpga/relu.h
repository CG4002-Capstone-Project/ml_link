#include "types.h"

template <
	int dim1,
	int dim2,
	int dim3
>
void relu(d_t (&arr)[dim1][dim2][dim3]) {
	for (int i = 0; i < dim1; i++) {
		for (int j = 0; j < dim2; j++) {
#pragma HLS PIPELINE
			op: for (int k = 0; k < dim3; k++) {
				arr[i][j][k] = std::max(arr[i][j][k], (d_t) 0);
			}
		}
	}
}

template <
	int dim1,
	int dim2
>
void relu(d_t (&arr)[dim1][dim2]) {
	for (int i = 0; i < dim1; i++) {
		for (int j = 0; j < dim2; j++) {
#pragma HLS PIPELINE
			arr[i][j] = std::max(arr[i][j], (d_t) 0);
		}
	}
}
