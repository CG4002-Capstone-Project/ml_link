#include "types.h"
#include "dot_product.h"

template <
	int batch_size,
	int in_features,
	int out_features
>
void linear(
	d_t (&input)[batch_size][in_features],
	d_t (&output)[batch_size][out_features],
	d_t (&weights)[out_features][in_features],
	d_t (&bias)[out_features]
) {
	batch: for (int batch = 0; batch < batch_size; batch++) {
		dot_product: for (int i = 0; i < out_features; i++) {
			output[batch][i] = bias[i] + dot<in_features>(weights[i], input[batch]);
		}
	}
}
