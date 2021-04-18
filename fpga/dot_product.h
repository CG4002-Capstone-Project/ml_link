#include "types.h"

template<int len>
d_t dot(d_t (&a)[len], d_t (&b)[len]) {
	d_t sum = 0;

	for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE
		sum += a[i] * b[i];
	}

	return sum;
}
