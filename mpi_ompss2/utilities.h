#ifndef _UTILITIES_H_
#define _UTILITIES_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <mpi.h>

#define ROOT 0
#define NUM_ADJ_PART 8
#define NUM_ADJ_GRID 4

// Direction in the 2D space (full, particles)
enum part_direction {
	PART_DOWN_LEFT = 0,
	PART_DOWN = 1,
	PART_DOWN_RIGHT = 2,
	PART_LEFT = 3,
	PART_RIGHT = 4,
	PART_UP_LEFT = 5,
	PART_UP = 6,
	PART_UP_RIGHT = 7
};
// Calculate the opposite direction
#define OPPOSITE_DIR(dir) ((NUM_ADJ_PART - 1) - dir)

// Direction in the 2D space (no diagonals, grid)
// The each fraction of the gaspi segment contains all ghost cells
// in a given direction
enum grid_direction {
	GRID_DOWN = 0,
	GRID_LEFT = 1,
	GRID_RIGHT = 2,
	GRID_UP = 3
};

enum mpi_tag {
	MPI_TAG_J = 0,
	MPI_TAG_B = 1,
	MPI_TAG_E = 2
};
#define MPI_TAG_PART(spec_id) (3 + spec_id)

// Calculate the notification ID based on the direction, region ID and a modifier
#define CREATE_MPI_TAG(dir, region_id, mod) (10000 * (mod) + 10 * (region_id) + (dir))

// Check for errors during the MPI routines
#define CHECK_MPI_ERROR(...)																		\
	do																								\
	{																								\
		const int r = __VA_ARGS__;														\
																									\
		if (r != MPI_SUCCESS)																		\
		{																							\
			printf ("Error: '%s' [%s:%i]: %i \n", #__VA_ARGS__, __FILE__, __LINE__, r);				\
			exit (EXIT_FAILURE);																	\
		}																							\
	} while (0)

// Enforce periodic boundaries
#define PERIODIC_BOUNDARIES(x, max) (x < 0 ? x + max : (x >= max ? x - max : x))
#define MAX_VALUE(x, y) (x > y ? x : y)
#define MIN_VALUE(x, y) (x < y ? x : y)

void get_optimal_division(int *div, int n);
void realloc_vector(void **restrict ptr, const int old_size, const int new_size, const size_t type_size);

#endif /* _UTILITIES_H_ */
