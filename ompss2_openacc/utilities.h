#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// NVIDIA GPUs
#define DEVICE_TYPE acc_device_nvidia

#define MAX_VALUE(x, y) x > y ? x : y
#define MIN_VALUE(x, y) x < y ? x : y
#define LTRIM(x) (x >= 1.0f) - (x < 0.0f)

#define LOCAL_BUFFER_SIZE 1024
#define TILE_SIZE 16
#define MIN_WARP_SIZE 32

void realloc_device_buffer(void **restrict ptr, const size_t old_size, const size_t new_size,
		const size_t type_size, const int device);
void* alloc_device_buffer(const size_t size, const int device);
void free_device_buffer(void *ptr);

void prefix_sum_openacc(int *restrict vector, const int size);
void prefix_sum_serial(int *restrict vector, const int size);

#ifdef ENABLE_PREFETCH
void grid_prefetch_openacc(t_vfld *buffer, const int size, const int device, void *stream);
#endif

#endif /* UTILITIES_H_ */
