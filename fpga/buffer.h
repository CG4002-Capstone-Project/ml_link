#include "ap_int.h"
#include "types.h"

void realign_buffer(
	d_t (&sensor_data_buffer)[N_DANCERS][N_SAMPLES][N_CHANNELS],
	d_t (&sensor_data)[N_DANCERS][N_SAMPLES][N_CHANNELS],
	int (&sensor_data_index)[N_DANCERS],
	dbg_t &dbg
) {
	for (int i = 0; i < N_DANCERS; i++) {
		for (int j = 0; j < N_SAMPLES; j++) {
			for (int k = 0; k < N_CHANNELS; k++) {
#pragma HLS PIPELINE
				int mapped_ind = (j + sensor_data_index[i]);
				if (mapped_ind >= N_SAMPLES) {
					mapped_ind -= N_SAMPLES;
				}
				sensor_data[i][j][k] =
						sensor_data_buffer[i][mapped_ind][k];
#ifdef DEBUG
				dbg.sensor_data[i][j][k] =
						sensor_data_buffer[i][mapped_ind][k];
#endif
			}
		}
	}
}
