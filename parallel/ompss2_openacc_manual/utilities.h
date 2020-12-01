#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_ASYNC_QUEUES 1
#define DEFAULT_ALIGNMENT 1 << 12

void realloc_align_buffer(void **restrict ptr, const size_t old_size, const size_t new_size,
		const size_t type_size, const size_t alignment);
void* alloc_align_buffer(const size_t alignment, const size_t size);
void free_align_buffer(void *ptr);

int get_gpu_async_queue(const int device);

#endif /* UTILITIES_H_ */
