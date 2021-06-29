#include "utilities.h"

/*********************************************************************************************
 Memory Management
 *********************************************************************************************/

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


inline void* alloc_device_buffer(const size_t size, const int device)
{

#ifdef ENABLE_AFFINITY
	return nanos6_device_alloc(size, device);
#else
	return malloc(size);
#endif
}

inline void free_device_buffer(void *ptr)
{
#ifdef ENABLE_AFFINITY
	nanos6_device_free(ptr);
#else
	free(ptr);
#endif
}

#ifdef ENABLE_PREFETCH
void grid_prefetch_openacc(t_vfld *buffer, const int size, const int device, void *stream)
{
	cuMemPrefetchAsync(buffer, size * sizeof(t_vfld), device, stream);
}
#endif

/*********************************************************************************************
 Prefix Sum
 *********************************************************************************************/

// Add the block offset to the vector
void add_block_sum(int *restrict vector, int *restrict block_sum, const int num_blocks,
        const int block_size, const int vector_size)
{
	#pragma acc parallel loop gang
	for (int block_id = 1; block_id < num_blocks; block_id++)
	{
		const int begin_idx = block_id * block_size;

		#pragma acc loop vector
		for (int i = 0; i < block_size; i++)
			if (i + begin_idx < vector_size) vector[i + begin_idx] += block_sum[block_id];
	}
}

// Prefix Sum (Exclusive) - 1 warp per thread block
void prefix_sum_min(int *restrict vector, int *restrict block_sum, const int num_blocks,
        const int size)
{
	// Prefix sum using a binomial tree
	#pragma acc parallel loop gang vector_length(MIN_WARP_SIZE)
	for (int block_id = 0; block_id < num_blocks; block_id++)
	{
		const int begin_idx = block_id * MIN_WARP_SIZE;
		int local_buffer[MIN_WARP_SIZE];

		#pragma acc cache(local_buffer[0: MIN_WARP_SIZE])

		// Copy to the local buffer
		#pragma acc loop vector
		for (int i = 0; i < MIN_WARP_SIZE; i++)
		{
			if (i + begin_idx < size) local_buffer[i] = vector[i + begin_idx];
			else local_buffer[i] = 0;
		}

		// Scan the tree upward (in direction to the root).
		// Add the values of each node and stores the result in the right node
		for (int offset = 1; offset < MIN_WARP_SIZE; offset *= 2)
		{
			#pragma acc loop vector
			for (int i = offset - 1; i < MIN_WARP_SIZE; i += 2 * offset)
				local_buffer[i + offset] += local_buffer[i];
		}

		// Store the total sum in the block sum vector and reset the last position of the vector
		block_sum[block_id] = local_buffer[MIN_WARP_SIZE - 1];
		local_buffer[MIN_WARP_SIZE - 1] = 0;

		// Scan the tree downward (from the root).
		// First, swap the values between nodes, then update the right node with the sum
		for (int offset = MIN_WARP_SIZE >> 1; offset > 0; offset >>= 1)
		{
			#pragma acc loop vector
			for (int i = offset - 1; i < MIN_WARP_SIZE; i += 2 * offset)
			{
				int temp = local_buffer[i];
				local_buffer[i] = local_buffer[i + offset];
				local_buffer[i + offset] += temp;
			}
		}

		// Store the results in the global vector
		#pragma acc loop vector
		for (int i = 0; i < MIN_WARP_SIZE; i++)
			if (i + begin_idx < size) vector[i + begin_idx] = local_buffer[i];
	}
}

// Prefix Sum (Exclusive) - Multiple warps per thread block
void prefix_sum_full(int *restrict vector, int *restrict block_sum, const int num_blocks,
        const int size)
{
	// Prefix sum using a binomial tree
	#pragma acc parallel loop gang vector_length(LOCAL_BUFFER_SIZE / 2) //
	for (int block_id = 0; block_id < num_blocks; block_id++)
	{
		const int begin_idx = block_id * LOCAL_BUFFER_SIZE;
		int local_buffer[LOCAL_BUFFER_SIZE];

		#pragma acc cache(local_buffer[0: LOCAL_BUFFER_SIZE])

		// Copy to the local buffer
		#pragma acc loop vector
		for (int i = 0; i < LOCAL_BUFFER_SIZE; i++)
		{
			if (i + begin_idx < size) local_buffer[i] = vector[i + begin_idx];
			else local_buffer[i] = 0;
		}

		// Scan the tree upward (in direction to the root).
		// Add the values of each node and stores the result in the right node
		for (int offset = 1; offset < LOCAL_BUFFER_SIZE; offset *= 2)
		{
			#pragma acc loop vector
			for (int i = offset - 1; i < LOCAL_BUFFER_SIZE; i += 2 * offset)
				local_buffer[i + offset] += local_buffer[i];

		}

		// Store the total sum in the block sum vector and reset the last position of the vector
		block_sum[block_id] = local_buffer[LOCAL_BUFFER_SIZE - 1];
		local_buffer[LOCAL_BUFFER_SIZE - 1] = 0;

		// Scan the tree downward (from the root).
		// First, swap the values between nodes, then update the right node with the sum
		for (int offset = LOCAL_BUFFER_SIZE >> 1; offset > 0; offset >>= 1)
		{
			#pragma acc loop vector
			for (int i = offset - 1; i < LOCAL_BUFFER_SIZE; i += 2 * offset)
			{
				int temp = local_buffer[i];
				local_buffer[i] = local_buffer[i + offset];
				local_buffer[i + offset] += temp;
			}
		}

		// Store the results in the global vector
		#pragma acc loop vector
		for (int i = 0; i < LOCAL_BUFFER_SIZE; i++)
			if (i + begin_idx < size) vector[i + begin_idx] = local_buffer[i];
	}
}

// Prefix/Scan Sum (Exclusive)
void prefix_sum_openacc(int *restrict vector, const int size)
{
	int num_blocks;
	int *restrict block_sum;

	if(size < LOCAL_BUFFER_SIZE / 4)
	{
		num_blocks = ceil((float) size / MIN_WARP_SIZE);
		block_sum = malloc(num_blocks * sizeof(int));

		prefix_sum_min(vector, block_sum, num_blocks, size);

		if (num_blocks > 1)
		{
			prefix_sum_openacc(block_sum, num_blocks);

			// Add the values from the block sum
			add_block_sum(vector, block_sum, num_blocks, MIN_WARP_SIZE, size);
		}
	} else
	{
		num_blocks = ceil((float) size / LOCAL_BUFFER_SIZE);
		block_sum = malloc(num_blocks * sizeof(int));

		prefix_sum_full(vector, block_sum, num_blocks, size);

		if (num_blocks > 1)
		{
			prefix_sum_openacc(block_sum, num_blocks);

			// Add the values from the block sum
			add_block_sum(vector, block_sum, num_blocks, LOCAL_BUFFER_SIZE, size);
		}
	}

	free(block_sum);
}

void prefix_sum_serial(int *restrict vector, const int size)
{
	int acc = 0;
	for (int i = 0; i < size; i++)
	{
		int temp = vector[i];
		vector[i] = acc;
		acc += temp;
	}
}
