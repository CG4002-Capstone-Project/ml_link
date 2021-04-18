#include "types.h"

template <
	int batch_size,
	int sample_size,
	int n_in_channels,
	int n_kernels,
	int kernel_width
>
void conv1d(
	d_t (&input)[batch_size][sample_size][n_in_channels],
	d_t (&kernel_wt)[n_kernels][n_in_channels][kernel_width],
	d_t (&bias)[n_kernels][sample_size - kernel_width + 1],
	d_t (&output)[batch_size][sample_size - kernel_width + 1][n_kernels]
) {
	batch: for (int batch = 0; batch < batch_size; batch++) {
		for (int k = 0; k < n_kernels; k++) {
			dot_product: for (int offset = 0; offset < sample_size - kernel_width + 1; offset++) {
				// perform multiply and accumulate in this window
				// starting from offset to offset - kernel_width + 1
				#pragma HLS PIPELINE

				d_t &sum = output[batch][offset][k];
				sum = bias[k][offset];

				for (int i = 0; i < kernel_width; i++) {
					for (int j = 0; j < n_in_channels; j++) {
						sum += input[batch][i + offset][j] * kernel_wt[k][j][i];
					}
				}
			}
		}
	}
}
