#ifndef _UTILITIES_H_
#define _UTILITIES_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <GASPI.h>

#define ROOT 0
#define NUM_ADJ_PART 8
#define NUM_ADJ_GRID 4
#define NUM_GASPI_QUEUES 16
#define DEFAULT_QUEUE 0

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

// Communication type (to determine the target zone in the gaspi segment)
enum gaspi_communication {
	GASPI_RECV = 0,
	GASPI_SEND = 1
};

// Get the segment offset depending on the operation (send or receive)
#define SEGM_OFFSET_PART(dir, op) (dir + op * 8)
#define SEGM_OFFSET_GRID(dir, op) (dir + op * 4)

enum gaspi_segment_id {
	E_SEGMENT_ID = 0,
	B_SEGMENT_ID = 1,
	J_SEGMENT_ID = 2
};

// Calculate the segment id based on the id of particle species
#define PART_SEGMENT_ID(spec_id) (3 + spec_id)

enum gaspi_notif_id {
	NOTIF_ID_CURRENT = 0,
	NOTIF_ID_CURRENT_ACK = 1,
	NOTIF_ID_EMF = 2,
	NOTIF_ID_EMF_ACK = 3
};

#define NOTIF_ID_PART(spec_id) (4 + 2 * spec_id)
#define NOTIF_ID_PART_ACK(spec_id) (5 + 2 * spec_id)

// Calculate the notification ID based on the direction, region ID and a modifier
#define NOTIFICATION_ID(dir, region_id, mod) (2000 * (mod) + 10 * (region_id) + (dir))

// Codes for the notification
enum gaspi_comm_codes {
	COMM_PART_WRITE = 1,
	COMM_PART_READ = 2,
	COMM_PART_ACK = 3,
	COMM_PART_PING = 4,
	COMM_CURRENT_WRITE = 11,
	COMM_CURRENT_READ = 12,
	COMM_CURRENT_ACK = 13,
	COMM_CURRENT_PING = 14,
	COMM_EMF_WRITE = 21,
	COMM_EMF_READ = 22,
	COMM_EMF_ACK = 23,
	COMM_EMF_PING = 24
};

// Check for errors during the GASPI routines
#define CHECK_GASPI_ERROR(...)																		\
	do																								\
	{																								\
		const gaspi_return_t r = __VA_ARGS__;														\
																									\
		if (r != GASPI_SUCCESS)																		\
		{																							\
			char* error = malloc(128);																\
			gaspi_print_error(r, &error);															\
			printf ("Error: '%s' [%s:%i]: %i => %s\n", #__VA_ARGS__, __FILE__, __LINE__, r, error);	\
			free(error);																			\
			exit (EXIT_FAILURE);																	\
		}																							\
	} while (0)

// Check for errors or timeout during the GASPI routines
#define CHECK_GASPI_TIMEOUT(...)																	\
	do																								\
	{																								\
		const gaspi_return_t r = __VA_ARGS__;														\
																									\
		if (r != GASPI_SUCCESS && r != GASPI_TIMEOUT)												\
		{																							\
			char* error = malloc(128);																\
			gaspi_print_error(r, &error);															\
			printf ("Error: '%s' [%s:%i]: %i => %s\n", #__VA_ARGS__, __FILE__, __LINE__, r, error);	\
			free(error);																			\
			exit (EXIT_FAILURE);																	\
		}																							\
	} while (0)

// Enforce periodic boundaries
#define PERIODIC_BOUNDARIES(x, max) (x < 0 ? x + max : (x >= max ? x - max : x))
#define MAX_VALUE(x, y) (x > y ? x : y)
#define MIN_VALUE(x, y) (x < y ? x : y)

void get_optimal_division(int *div, int n);
void realloc_vector(void **restrict ptr, const int old_size, const int new_size, const size_t type_size);

unsigned int get_gaspi_queue(const unsigned int region_id);
void gaspi_flush_all_queues();
void gaspi_reduce_float(float *buf, const size_t buf_size, const gaspi_rank_t root,
                        const gaspi_group_t group);
bool gaspi_notify_test(const gaspi_segment_id_t segm_id, const gaspi_notification_id_t notif_id);
void gaspi_recv(const gaspi_segment_id_t segm_id, const int notif_ids[8],
                      const gaspi_notification_t expected);

#endif /* _UTILITIES_H_ */
