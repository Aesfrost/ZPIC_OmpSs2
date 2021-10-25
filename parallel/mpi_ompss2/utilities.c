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

void mpi_wait_async_comm(MPI_Request *requests, const unsigned int num_requests)
{
	if(num_requests > 0 && requests)
	{

#ifdef ENABLE_TASKING
		int flag;
		CHECK_MPI_ERROR(MPI_Testall(num_requests, requests, &flag, MPI_STATUSES_IGNORE));
		if(!flag) block_comm_task(requests, num_requests);
#else
		CHECK_MPI_ERROR(MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE));
#endif

	}
}
