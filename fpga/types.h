#ifndef TYPES_H
#define TYPES_H

#include "ap_int.h"
#include "ap_fixed.h"

typedef ap_fixed<32, 8> d_t;

typedef struct {
#ifdef DEBUG
	d_t sensor_data[N_DANCERS][N_SAMPLES][N_CHANNELS];
	d_t layer1_out[N_DANCERS][N_SAMPLES - CONV1_KERNEL_WIDTH + 1][CONV1_N_KERNELS];
	d_t flattened[N_DANCERS][(N_SAMPLES - CONV1_KERNEL_WIDTH + 1)/2 * CONV1_N_KERNELS];
	d_t maxpool[N_DANCERS][(N_SAMPLES - CONV1_KERNEL_WIDTH + 1)/2][CONV1_N_KERNELS];
	d_t fc1_output[N_DANCERS][FC1_N_OUT];
#endif
} dbg_t;

typedef struct {
	// fc1
	d_t fc1_wt[FC1_OUT][FC1_IN];
	d_t fc1_bias[FC1_OUT];

	// fc2
	d_t fc2_wt[FC2_OUT][FC1_OUT];
	d_t fc2_bias[FC2_OUT];

	// fc3
	d_t fc3_wt[FC3_OUT][FC2_OUT];
	d_t fc3_bias[FC3_OUT];
} weights_t;

typedef struct {
	d_t result[N_DANCERS][FC3_OUT];
} result_t;

#endif
