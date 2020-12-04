#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void realloc_device_buffer(void **restrict ptr, const size_t old_size, const size_t new_size,
		const size_t type_size, const int device);
void* alloc_device_buffer(const size_t size, const int device);
void free_device_buffer(void *ptr);

#endif /* UTILITIES_H_ */
