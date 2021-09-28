#include "utilities.h"

// Calculate the optimal decomposition of the a number n in Cartesian coordinates
void get_optimal_division(int *div, int n)
{
	int rt = sqrtf(n);

	if(rt * rt == n) // integer square root
	{
		div[0] = rt;
		div[1] = rt;
	}else
	{
		int factors[rt]; // Factors
		int num_factors = 0;

		div[0] = 1;
		div[1] = 1;

		// Find all factors "2"
		while(n % 2 == 0 && n > 1)
		{
			factors[num_factors++] = 2;
			n /= 2;
		}

		// Find all factors "3"
		while(n % 3 == 0 && n > 1)
		{
			factors[num_factors++] = 3;
			n /= 3;
		}

		// Find all factors "5"
		while(n % 5 == 0 && n > 1)
		{
			factors[num_factors++] = 5;
			n /= 5;
		}

		// Add the result as a factor
		if(n > 1)
			factors[num_factors++] = n;

		// Try to distribute the factors more evenly as possible
		for(int i = num_factors - 1; i >= 0; i--)
		{
			if(div[0] * factors[i] <= rt)
				div[0] *= factors[i];
			else
				div[1] *= factors[i];
		}
	}
}

// Manual reallocation of buffers
void realloc_vector(void **restrict ptr, const int old_size, const int new_size,
                    const size_t type_size)
{
//	#pragma acc set device_num(0) // Dummy operation to work with the PGI Compiler

	if(*ptr == NULL) *ptr = malloc(new_size * type_size);
	else
	{
		void *restrict temp = malloc(new_size * type_size);

		if(temp)
		{
			memcpy(temp, *ptr, old_size * type_size);
			free(*ptr);
			*ptr = temp;
		}else
		{
			printf("Error in allocating particle vector. Exiting...\n");
			exit(1);
		}
	}
}

bool gaspi_notify_test(const gaspi_segment_id_t segm_id, const gaspi_notification_id_t notif_id)
{
	gaspi_notification_id_t id;
	return (gaspi_notify_waitsome(segm_id, notif_id, 1, &id, GASPI_TEST) == GASPI_SUCCESS);
}

void gaspi_recv(const gaspi_segment_id_t segm_id, const int notif_ids[8])
{
	gaspi_notification_t value;

	for (int i = 0; i < 8; ++i)
		if(notif_ids[i] >= 0)
			CHECK_GASPI_ERROR(tagaspi_notify_async_wait(segm_id, notif_ids[i], &value));
}

// Get a gaspi queue from the pool
unsigned int get_gaspi_queue(const unsigned int region_id)
{
	unsigned int queue = region_id % NUM_GASPI_QUEUES;
	unsigned int queue_size;
	unsigned int queue_size_max;

	CHECK_GASPI_ERROR(gaspi_queue_size(queue, &queue_size));
	CHECK_GASPI_ERROR(gaspi_queue_size_max(&queue_size_max));

	if(queue_size > (float) 0.9f * queue_size_max)
		CHECK_GASPI_ERROR(gaspi_wait(queue, GASPI_BLOCK));

	return queue;
}

void gaspi_flush_all_queues()
{
	for (int queue = 0; queue < NUM_GASPI_QUEUES; ++queue)
		CHECK_GASPI_ERROR(gaspi_wait(queue, GASPI_BLOCK));
}
