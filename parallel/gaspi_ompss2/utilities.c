#include "utilities.h"
#include "task_management.h"

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

void gaspi_recv(const gaspi_segment_id_t segm_id, const int notif_ids[8],
				const gaspi_notification_t expected)
{
	gaspi_notification_t value;
	bool received = true;

#ifdef ENABLE_TASKING

	for (int i = 0; i < 8; ++i)
		if(notif_ids[i] >= 0)
			received &= gaspi_notify_test(segm_id, notif_ids[i]);

	if(!received) block_comm_task(segm_id, notif_ids);

#else
	gaspi_notification_id_t id;
	for (int i = 0; i < 8; ++i)
		if(notif_ids[i] >= 0)
			CHECK_GASPI_ERROR(gaspi_notify_waitsome(segm_id, notif_ids[i], 1, &id, GASPI_BLOCK));

#endif

	for (int i = 0; i < 8; ++i)
	{
		if(notif_ids[i] >= 0)
		{
			CHECK_GASPI_ERROR(gaspi_notify_reset(segm_id, notif_ids[i], &value));

			if(value != expected)
			{
				fprintf(stderr, "Error: Wrong notification received. Segm ID: %d | ID: %d | "
						"Value: %d | Expected: %d\n", segm_id, notif_ids[i], value, expected);
			}
		}
	}
}

// Perform a reduction operation (operation: sum) in a buffer of floats
#define REDUCE_ID 99
void gaspi_reduce_float(float *buf, const size_t buf_size, const gaspi_rank_t root,
                        const gaspi_group_t group)
{
	gaspi_notification_id_t id;
	gaspi_notification_t value;

	gaspi_rank_t num_proc, rank;
	CHECK_GASPI_ERROR(gaspi_proc_rank(&rank));
	CHECK_GASPI_ERROR(gaspi_proc_num(&num_proc));

	gaspi_pointer_t ptr;
	CHECK_GASPI_ERROR(gaspi_segment_create(REDUCE_ID, buf_size * sizeof(float),
	                                       group, GASPI_BLOCK, GASPI_MEM_INITIALIZED));
	CHECK_GASPI_ERROR(gaspi_segment_ptr(REDUCE_ID, &ptr));
	float *restrict gaspi_segm = (float *) ptr;

	if(rank == root)
	{
		for(int i = 0; i < num_proc; i++)
		{
			if(i != root)
			{
				CHECK_GASPI_ERROR(gaspi_notify_waitsome(REDUCE_ID, i, 1, &id, GASPI_BLOCK));
				CHECK_GASPI_ERROR(gaspi_notify_reset(REDUCE_ID, id, &value));
				CHECK_GASPI_ERROR(gaspi_read(REDUCE_ID, 0, i, REDUCE_ID, 0,
				                             buf_size * sizeof(float), DEFAULT_QUEUE, GASPI_BLOCK));
				CHECK_GASPI_ERROR(gaspi_wait(DEFAULT_QUEUE, GASPI_BLOCK));

				for (int k = 0; k < buf_size; ++k)
					buf[k] += gaspi_segm[k];
			}
		}
	}else
	{
		memcpy(gaspi_segm, buf, buf_size * sizeof(float));
		CHECK_GASPI_ERROR(gaspi_notify(REDUCE_ID, root, rank, REDUCE_ID, DEFAULT_QUEUE, GASPI_BLOCK));
	}

	CHECK_GASPI_ERROR(gaspi_barrier(group, GASPI_BLOCK));
	CHECK_GASPI_ERROR(gaspi_segment_delete(REDUCE_ID));
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
