#include "utilities.h"

// Manual reallocation of a buffer aligned in the memory
void realloc_device_buffer(void **restrict ptr, const size_t old_size, const size_t new_size,
		const size_t type_size, const int device)
{
	void *temp = alloc_device_buffer(new_size * type_size, device); // Allocate new buffer

	if(temp)
	{
		memcpy(temp, *ptr, old_size * type_size); // Copy the data from the old buffer to the new one
		free_device_buffer(*ptr); // Free the old buffer
		*ptr = temp; // Point to the new buffer
	}else
	{
		printf("Error in allocating vector. Exiting...\n");
		exit(1);
	}
}


void* alloc_device_buffer(const size_t size, const int device)
{

#ifdef ENABLE_AFFINITY
	return nanos6_device_alloc(size, device);
#else
	return malloc(size);
#endif
}

void free_device_buffer(void *ptr)
{
#ifdef ENABLE_AFFINITY
	nanos6_device_free(ptr);
#else
	free(ptr);
#endif
}
