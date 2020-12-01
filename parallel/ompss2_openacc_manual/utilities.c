#include "utilities.h"

static int _gpu_async_queue[4] = {0};

// Manual reallocation of a buffer aligned in the memory
void realloc_align_buffer(void **restrict ptr, const size_t old_size, const size_t new_size,
		const size_t type_size, const size_t alignment)
{
	void *temp = alloc_align_buffer(alignment, new_size * type_size); // Allocate new buffer

	if(temp)
	{
		memcpy(temp, *ptr, old_size * type_size); // Copy the data from the old buffer to the new one
		free_align_buffer(*ptr); // Free the old buffer
		*ptr = temp; // Point to the new buffer
	}else
	{
		printf("Error in allocating particle vector. Exiting...\n");
		exit(1);
	}
}

// Manual allocation of a buffer aligned in the memory
void* alloc_align_buffer(const size_t alignment, const size_t size)
{
	#pragma acc set device_num(0)

	return malloc(size * sizeof(char));

	// Total size of the buffer (size of the data + padding + address of the beginning of the buffer)
	const size_t total_size = size + alignment + sizeof(size_t);
	void *ptr = malloc(total_size * sizeof(char)); // Allocate the buffer

	if(ptr)
	{
		const void *const ptr_start = ptr; // Stores the address of the beginning of the buffer
		ptr += sizeof(size_t);

		const size_t offset = alignment - (((size_t) ptr) % alignment); // Add the padding to the buffer
		ptr += offset;

		size_t *book_keeping = (size_t*)(ptr - sizeof(size_t)); // Stores the true beginning of the buffer
		*book_keeping = (size_t) ptr_start;

	}else
	{
		printf("Error in allocating buffer... Exiting\n");
		exit(1);
	}

	return ptr;
}

// Manual free of a buffer aligned in the memory
void free_align_buffer(void *ptr)
{
	free(ptr);
	return;

	if(ptr)
	{
		// Recover the original address
		char *addr = ptr;
		addr -= sizeof(size_t);
		addr = (char*)(*(size_t*) addr);

		free(addr);
	}
}

int get_gpu_async_queue(const int device)
{
	int queue;

	#pragma acc atomic capture
	queue = _gpu_async_queue[device]++;
	queue = queue % MAX_ASYNC_QUEUES;

	return queue;
}
