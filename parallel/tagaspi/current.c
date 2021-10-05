/*********************************************************************************************
 ZPIC
 current.c

 Created by Ricardo Fonseca on 12/8/10.
 Modified by Nicolas Guidotti on 11/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#include "current.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "utilities.h"
#include "zdf.h"

/*********************************************************************************************
 Constructor / Destructor
 *********************************************************************************************/
void current_new(t_current *current, int nx[], t_fld box[], float dt, bool on_right_edge,
                 bool on_left_edge)
{
	int i;

	// Number of guard cells for linear interpolation
	int gc[2][2] = { {1, 2}, {1, 2}};

	current->nrow = gc[0][0] + nx[0] + gc[0][1];
	current->ncol = gc[1][0] + nx[1] + gc[1][1];

	// Allocate global array
	size_t size = current->ncol * current->nrow;
	current->total_size = size;
	current->overlap_size = current->nrow * (gc[1][0] + gc[1][1]);

	current->J_buf = calloc(size, sizeof(t_vfld));
	assert(current->J_buf);

	// store nx and gc values
	for (i = 0; i < 2; i++)
	{
		current->nx[i] = nx[i];
		current->gc[i][0] = gc[i][0];
		current->gc[i][1] = gc[i][1];
	}

	// Make J point to cell [0][0]
	current->J = current->J_buf + gc[0][0] + gc[1][0] * current->nrow;

	// Set cell sizes and box limits
	for (i = 0; i < 2; i++)
	{
		current->box[i] = box[i];
		current->dx[i] = box[i] / nx[i];
	}

	// Clear smoothing options
	current->smooth = (t_smooth) {.xtype = NONE, .ytype = NONE, .xlevel = 0, .ylevel = 0};

	// Initialize time information
	current->iter = 0;
	current->dt = dt;

	current->moving_window = false;

	current->first_comm = true;

	current->on_left_edge = on_left_edge;
	current->on_right_edge = on_right_edge;

	for (int i = 0; i < 4; ++i)
		current->gaspi_notif[i] = 0;
}

void current_delete(t_current *current)
{
	free(current->J_buf);
	current->J_buf = NULL;
}

// Set the current buffer to zero
void current_zero(t_current *current)
{
	current->iter++;
	printf("Iter: %d\n", current->iter);

	// zero fields
	size_t size = current->nrow * current->ncol;
	memset(current->J_buf, 0, size * sizeof(t_vfld));
}

/*********************************************************************************************
 Communication
 *********************************************************************************************/

void current_link_adj_regions(t_current *current, t_current *current_down, t_current *current_up,
                              t_vfld *gaspi_segm_J, const int segm_offset[2 * NUM_ADJ_GRID],
                              const gaspi_rank_t adj_ranks[NUM_ADJ_GRID], const int region_id,
                              const int region_limits[2][2], const int proc_limits[2][2])
{
	const int segm_nrow = current->gc[0][0] + current->gc[0][1];
	const int segm_ncol = current->gc[1][0] + current->gc[1][1];

	// Offset to the beginning of the region (remember that region limits are in global coordinates)
	const int offset_y = segm_nrow * (current->gc[1][0] + region_id * segm_ncol
					   + (region_limits[1][0] - proc_limits[1][0]));

	for (int dir = 0; dir < 4; dir++)
	{
		int notif_id = -1;
		current->gaspi_segm_offset_send[dir] = segm_offset[SEGM_OFFSET_GRID(dir, GASPI_SEND)];
		current->gaspi_segm_offset_recv[dir] = segm_offset[SEGM_OFFSET_GRID(dir, GASPI_RECV)];

		switch (dir)
		{
			case GRID_DOWN:
				if (current_down)   // The adjacent region (bottom) is in the same process
				{
					current->send_J[dir] = current_down->J_buf
					        + current_down->nx[1] * current_down->nrow;
					current->receive_J[dir] = current->send_J[dir];

					current->gaspi_segm_offset_recv[dir] = -1;   // Local communication
					current->gaspi_segm_offset_send[dir] = -1;   // Local communication

				} else notif_id = NOTIFICATION_ID(GRID_UP, 0, 0);
				break;

			case GRID_UP:
				if (current_up)   // The adjacent region (upper) is in the same process
				{
					// Ghost cells in the left
					current->send_J[dir] = current_up->J_buf;
					current->receive_J[dir] = current->send_J[dir];

					current->gaspi_segm_offset_recv[dir] = -1;   // Local communication
					current->gaspi_segm_offset_send[dir] = -1;   // Local communication

				} else notif_id = NOTIFICATION_ID(GRID_DOWN, 0, 0);
				break;

			default:   // GRID_LEFT or GRID_RIGHT

				current->gaspi_segm_offset_send[dir] += offset_y;
				current->gaspi_segm_offset_recv[dir] += offset_y;
				notif_id = NOTIFICATION_ID(3 - dir, region_id, 0);
				break;
		}

		if (notif_id >= 0)
		{
			current->send_J[dir] = &gaspi_segm_J[current->gaspi_segm_offset_send[dir]];
			current->receive_J[dir] = &gaspi_segm_J[current->gaspi_segm_offset_recv[dir]];

			MPI_Isend(&current->gaspi_segm_offset_recv[dir], 1, MPI_INT, adj_ranks[dir], notif_id,
			          MPI_COMM_WORLD, &current->requests[2 * dir]);

			if (dir == GRID_LEFT || dir == GRID_RIGHT) notif_id = NOTIFICATION_ID(dir, region_id, 0);
			else notif_id = NOTIFICATION_ID(dir, 0, 0);

			MPI_Irecv(&current->gaspi_remote_offset_send[dir], 1, MPI_INT, adj_ranks[dir], notif_id,
			          MPI_COMM_WORLD, &current->requests[2 * dir + 1]);
		}else
		{
			current->requests[2 * dir] = MPI_REQUEST_NULL;
			current->requests[2 * dir + 1] = MPI_REQUEST_NULL;
		}
	}
}

void current_comm_wait(t_current *current)
{
	MPI_Waitall(2 * NUM_ADJ_GRID, current->requests, MPI_STATUSES_IGNORE);
}

void current_wait_comm_x(t_current *current, const int region_id, const int notif_mod)
{
	if (!current->first_comm)
	{
		if (!current->moving_window || !current->on_left_edge)
		{
			int id = NOTIFICATION_ID(GRID_LEFT, region_id, notif_mod);
			CHECK_GASPI_ERROR(tagaspi_notify_async_wait(J_SEGMENT_ID, id, &current->gaspi_notif[GRID_LEFT]));
		}

		if (!current->moving_window || !current->on_right_edge)
		{
			int id = NOTIFICATION_ID(GRID_RIGHT, region_id, notif_mod);
			CHECK_GASPI_ERROR(tagaspi_notify_async_wait(J_SEGMENT_ID, id, &current->gaspi_notif[GRID_RIGHT]));
		}
	}
}

void current_wait_comm_y(t_current *current, const int notif_mod)
{
	if (notif_mod == NOTIF_ID_CURRENT_ACK && current->iter == 1) return;

	if (current->gaspi_segm_offset_send[GRID_UP] >= 0)
	{
		int id = NOTIFICATION_ID(GRID_UP, 0, notif_mod);
		CHECK_GASPI_ERROR(tagaspi_notify_async_wait(J_SEGMENT_ID, id, &current->gaspi_notif[GRID_UP]));
	}

	if (current->gaspi_segm_offset_send[GRID_DOWN] >= 0)
	{
		int id = NOTIFICATION_ID(GRID_DOWN, 0, notif_mod);
		CHECK_GASPI_ERROR(tagaspi_notify_async_wait(J_SEGMENT_ID, id, &current->gaspi_notif[GRID_DOWN]));
	}
}

void current_send_gc_x(t_current *current, const int region_id, const gaspi_rank_t adj_ranks[4])
{
	const unsigned int queue = get_gaspi_queue(region_id);

	const int nrow = current->nrow;
	t_vfld *restrict J = current->J;
	t_vfld *restrict J_left = current->send_J[GRID_LEFT];
	t_vfld *restrict J_right = current->send_J[GRID_RIGHT];

	const int segm_nrow = current->gc[0][0] + current->gc[0][1];
	int msg_size = segm_nrow * current->ncol;
	int offset_base = -current->gc[1][0] * segm_nrow;

	const int notif_id[4] = {0, NOTIFICATION_ID(GRID_LEFT, region_id, NOTIF_ID_CURRENT_ACK),
	                         NOTIFICATION_ID(GRID_RIGHT, region_id, NOTIF_ID_CURRENT_ACK), 0};
	check_notif_value(notif_id, current->gaspi_notif, COMM_CURRENT_ACK);

	if (!current->moving_window || !current->on_left_edge)
	{
		const int notif_id = NOTIFICATION_ID(GRID_RIGHT, region_id, NOTIF_ID_CURRENT);
		const int local_offset = offset_base + current->gaspi_segm_offset_send[GRID_LEFT];
		const int remote_offset = offset_base + current->gaspi_remote_offset_send[GRID_LEFT];

		for (int j = -current->gc[1][0]; j < current->nx[1] + current->gc[1][1]; ++j)
			for (int i = -current->gc[0][0]; i < current->gc[0][1]; ++i)
				J_left[i + current->gc[0][0] + j * segm_nrow] = J[i + j * nrow];

		CHECK_GASPI_ERROR(tagaspi_write_notify(J_SEGMENT_ID, 	// Local segment ID
		        local_offset * sizeof(t_vfld),				// Local segment offset
		        adj_ranks[GRID_LEFT],						// Rank of the receiving process
		        J_SEGMENT_ID,								// Remote segment ID
		        remote_offset * sizeof(t_vfld),				// Remote segment offset
		        msg_size * sizeof(t_vfld),					// Size
		        notif_id,									// Notification ID
		        COMM_CURRENT_WRITE,							// Notification value
		        queue));									// Queue
	}

	if (!current->moving_window || !current->on_right_edge)
	{
		const int notif_id = NOTIFICATION_ID(GRID_LEFT, region_id, NOTIF_ID_CURRENT);
		const int local_offset = offset_base + current->gaspi_segm_offset_send[GRID_RIGHT];
		const int remote_offset = offset_base + current->gaspi_remote_offset_send[GRID_RIGHT];

		for (int j = -current->gc[1][0]; j < current->nx[1] + current->gc[1][1]; ++j)
			for (int i = -current->gc[0][0]; i < current->gc[0][1]; ++i)
				J_right[i + current->gc[0][0] + j * segm_nrow] = J[current->nx[0] + i + j * nrow];

		CHECK_GASPI_ERROR(tagaspi_write_notify(J_SEGMENT_ID, 	// Local segment ID
		        local_offset * sizeof(t_vfld),				// Local segment offset
		        adj_ranks[GRID_RIGHT],						// Rank of the receiving process
		        J_SEGMENT_ID,								// Remote segment ID
		        remote_offset * sizeof(t_vfld),				// Remote segment offset
		        msg_size * sizeof(t_vfld),					// Size
		        notif_id,									// Notification ID
		        COMM_CURRENT_WRITE,							// Notification value
		        queue));									// Queue
	}

	current->first_comm = false;
}

void current_reduction_x(t_current *current, const int region_id, gaspi_rank_t adj_ranks[4])
{
	const unsigned int queue = get_gaspi_queue(region_id);

	const int nrow = current->nrow;
	const int segm_nrow = current->gc[0][0] + current->gc[0][1];

	t_vfld *restrict J = current->J;
	t_vfld *restrict J_left = current->receive_J[GRID_LEFT];
	t_vfld *restrict J_right = current->receive_J[GRID_RIGHT];

	const int notif_id[4] = {0, NOTIFICATION_ID(GRID_LEFT, region_id, NOTIF_ID_CURRENT),
	                         NOTIFICATION_ID(GRID_RIGHT, region_id, NOTIF_ID_CURRENT), 0};
	check_notif_value(notif_id, current->gaspi_notif, COMM_CURRENT_WRITE);

	if (!current->moving_window || !current->on_left_edge)
	{
		for (int j = -current->gc[1][0]; j < current->nx[1] + current->gc[1][1]; ++j)
		{
			for (int i = -current->gc[0][0]; i < current->gc[0][1]; ++i)
			{
				J[i + j * nrow].x += J_left[i + current->gc[0][0] + j * segm_nrow].x;
				J[i + j * nrow].y += J_left[i + current->gc[0][0] + j * segm_nrow].y;
				J[i + j * nrow].z += J_left[i + current->gc[0][0] + j * segm_nrow].z;
			}
		}

		int notif_id = NOTIFICATION_ID(GRID_RIGHT, region_id, NOTIF_ID_CURRENT_ACK);
		CHECK_GASPI_ERROR(tagaspi_notify(J_SEGMENT_ID, adj_ranks[GRID_LEFT], notif_id, COMM_CURRENT_ACK,
		                               queue));
	}

	if (!current->moving_window || !current->on_right_edge)
	{
		for (int j = -current->gc[1][0]; j < current->nx[1] + current->gc[1][1]; ++j)
		{
			for (int i = -current->gc[0][0]; i < current->gc[0][1]; ++i)
			{
				J[current->nx[0] + i + j * nrow].x += J_right[i + current->gc[0][0] + j * segm_nrow].x;
				J[current->nx[0] + i + j * nrow].y += J_right[i + current->gc[0][0] + j * segm_nrow].y;
				J[current->nx[0] + i + j * nrow].z += J_right[i + current->gc[0][0] + j * segm_nrow].z;
			}
		}

		int notif_id = NOTIFICATION_ID(GRID_LEFT, region_id, NOTIF_ID_CURRENT_ACK);
		CHECK_GASPI_ERROR(tagaspi_notify(J_SEGMENT_ID, adj_ranks[GRID_RIGHT], notif_id, COMM_CURRENT_ACK,
		                               queue));
	}
}

void current_update_gc_x(t_current *current, const int region_id, gaspi_rank_t adj_ranks[4])
{
	const unsigned int queue = get_gaspi_queue(region_id);

	const int nrow = current->nrow;
	const int segm_nrow = current->gc[0][0] + current->gc[0][1];

	t_vfld *restrict J = current->J;
	t_vfld *restrict J_left = current->receive_J[GRID_LEFT];
	t_vfld *restrict J_right = current->receive_J[GRID_RIGHT];

	const int notif_id[4] = {0, NOTIFICATION_ID(GRID_LEFT, region_id, NOTIF_ID_CURRENT),
	                         NOTIFICATION_ID(GRID_RIGHT, region_id, NOTIF_ID_CURRENT), 0};
	check_notif_value(notif_id, current->gaspi_notif, COMM_CURRENT_WRITE);

	if (!(current->moving_window && current->on_left_edge))
	{
		for (int j = -current->gc[1][0]; j < current->nx[1] + current->gc[1][1]; ++j)
			for (int i = -current->gc[0][0]; i < 0; ++i)
				J[i + j * nrow] = J_left[i + current->gc[0][0] + j * segm_nrow];

		int notif_id = NOTIFICATION_ID(GRID_RIGHT, region_id, NOTIF_ID_CURRENT_ACK);
		CHECK_GASPI_ERROR(tagaspi_notify(J_SEGMENT_ID, adj_ranks[GRID_LEFT], notif_id, COMM_CURRENT_ACK,
		                               queue));
	}

	if (!(current->moving_window && current->on_right_edge))
	{
		for (int j = -current->gc[1][0]; j < current->nx[1] + current->gc[1][1]; ++j)
			for (int i = 0; i < current->gc[0][1]; ++i)
				J[current->nx[0] + i + j * nrow] = J_right[i + current->gc[0][0] + j * segm_nrow];

		int notif_id = NOTIFICATION_ID(GRID_LEFT, region_id, NOTIF_ID_CURRENT_ACK);
		CHECK_GASPI_ERROR(tagaspi_notify(J_SEGMENT_ID, adj_ranks[GRID_RIGHT], notif_id, COMM_CURRENT_ACK,
		                               queue));
	}
}


void current_send_gc_y(t_current *current, const int region_id, const gaspi_rank_t adj_ranks[4])
{
	int remote_offset;
	const int nrow = current->nrow;

	const unsigned int queue = get_gaspi_queue(region_id);

	const int notif_id[4] = {NOTIFICATION_ID(GRID_DOWN, 0, NOTIF_ID_CURRENT_ACK), 0,
	                         0, NOTIFICATION_ID(GRID_UP, 0, NOTIF_ID_CURRENT_ACK)};
	check_notif_value(notif_id, current->gaspi_notif, COMM_CURRENT_ACK);

	if (current->gaspi_segm_offset_send[GRID_DOWN] >= 0)
	{
		remote_offset = current->gaspi_remote_offset_send[GRID_DOWN];
		memcpy(current->send_J[GRID_DOWN], current->J_buf, current->overlap_size * sizeof(t_vfld));

		CHECK_GASPI_ERROR(tagaspi_write_notify(J_SEGMENT_ID, 						// Local segment ID
		        current->gaspi_segm_offset_send[GRID_DOWN] * sizeof(t_vfld),	// Local segment offset
		        adj_ranks[GRID_DOWN],											// Rank of the receiving process
		        J_SEGMENT_ID,													// Remote segment ID
		        remote_offset * sizeof(t_vfld),									// Remote segment offset
		        current->overlap_size * sizeof(t_vfld),							// Size
		        NOTIFICATION_ID(GRID_UP, 0, NOTIF_ID_CURRENT),					// Notification ID
		        COMM_CURRENT_WRITE,												// Notification value
		        queue));														// Queue
	}

	if (current->gaspi_segm_offset_send[GRID_UP] >= 0)
	{
		remote_offset = current->gaspi_remote_offset_send[GRID_UP];
		memcpy(current->send_J[GRID_UP], current->J_buf + current->nx[1] * nrow,
		       current->overlap_size * sizeof(t_vfld));

		CHECK_GASPI_ERROR(tagaspi_write_notify(J_SEGMENT_ID, 						// Local segment ID
		        current->gaspi_segm_offset_send[GRID_UP] * sizeof(t_vfld),		// Local segment offset
		        adj_ranks[GRID_UP],												// Rank of the receiving process
		        J_SEGMENT_ID,													// Remote segment ID
		        remote_offset * sizeof(t_vfld),									// Remote segment offset
		        current->overlap_size * sizeof(t_vfld),							// Size
		        NOTIFICATION_ID(GRID_DOWN, 0, NOTIF_ID_CURRENT),				// Notification ID
		        COMM_CURRENT_WRITE,												// Notification value
		        queue));														// Queue
	}
}

// Each region is only responsible to do the reduction operation in its bottom edge
void current_reduction_y(t_current *current, const int region_id, const gaspi_rank_t adj_ranks[4])
{
	const unsigned int queue = get_gaspi_queue(region_id);

	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J_buf;
	t_vfld *restrict const J_down = current->receive_J[GRID_DOWN];

	const int notif_id[4] = {NOTIFICATION_ID(GRID_DOWN, 0, NOTIF_ID_CURRENT), 0,
	                         0, NOTIFICATION_ID(GRID_UP, 0, NOTIF_ID_CURRENT)};
	check_notif_value(notif_id, current->gaspi_notif, COMM_CURRENT_WRITE);

	for (int j = 0; j < current->gc[1][0] + current->gc[1][1]; j++)
	{
		for (int i = 0; i < current->nrow; i++)
		{
			J[i + j * nrow].x += J_down[i + j * nrow].x;
			J[i + j * nrow].y += J_down[i + j * nrow].y;
			J[i + j * nrow].z += J_down[i + j * nrow].z;

			J_down[i + j * nrow] = J[i + j * nrow];
		}
	}

	if (current->gaspi_segm_offset_recv[GRID_DOWN] >= 0)
	{
		int notif_id = NOTIFICATION_ID(GRID_UP, 0, NOTIF_ID_CURRENT_ACK);
		CHECK_GASPI_ERROR(tagaspi_notify(J_SEGMENT_ID, adj_ranks[GRID_DOWN], notif_id,
		                               COMM_CURRENT_ACK, queue));
	}

	if (current->gaspi_segm_offset_recv[GRID_UP] >= 0)
	{
		t_vfld *restrict const J_up = current->receive_J[GRID_UP];
		for (int j = 0; j < current->gc[1][0] + current->gc[1][1]; j++)
		{
			for (int i = 0; i < current->nrow; i++)
			{
				J[i + (j + current->nx[1]) * nrow].x += J_up[i + j * nrow].x;
				J[i + (j + current->nx[1]) * nrow].y += J_up[i + j * nrow].y;
				J[i + (j + current->nx[1]) * nrow].z += J_up[i + j * nrow].z;
			}
		}

		int notif_id = NOTIFICATION_ID(GRID_DOWN, 0, NOTIF_ID_CURRENT_ACK);
		CHECK_GASPI_ERROR(tagaspi_notify(J_SEGMENT_ID, adj_ranks[GRID_UP], notif_id,
		                               COMM_CURRENT_ACK, queue));
	}
}


/*********************************************************************************************
 Current Smoothing
 *********************************************************************************************/

// Gets the value of the compensator kernel for an n pass binomial kernel
void get_smooth_comp(int n, t_fld *sa, t_fld *sb)
{
	t_fld a, b, total;

	a = -1;
	b = (4.0 + 2.0 * n) / n;
	total = 2 * a + b;

	*sa = a / total;
	*sb = b / total;
}

// Apply the filter in the x direction
void kernel_x(t_current *const current, const t_fld sa, const t_fld sb)
{
	t_vfld *restrict const J = current->J;
	const int nrow = current->nrow;

	for (int j = -current->gc[1][0]; j < current->nx[1] + current->gc[1][1]; j++)
	{
		int idx = j * nrow;

		t_vfld fl = J[idx - 1];
		t_vfld f0 = J[idx];

		for (int i = 0; i < current->nx[0]; i++)
		{
			t_vfld fu = J[idx + i + 1];
			t_vfld fs;

			fs.x = sa * fl.x + sb * f0.x + sa * fu.x;
			fs.y = sa * fl.y + sb * f0.y + sa * fu.y;
			fs.z = sa * fl.z + sb * f0.z + sa * fu.z;

			J[idx + i] = fs;

			fl = f0;
			f0 = fu;
		}
	}
}

// Apply the filter in the y direction
void kernel_y(t_current *const current, const t_fld sa, const t_fld sb)
{
	t_vfld flbuf[current->nx[0]];
	t_vfld *restrict const J = current->J;
	const int nrow = current->nrow;

	int i, j;

	// buffer lower row
	for (i = 0; i < current->nx[0]; i++)
	{
		flbuf[i] = J[i - nrow];
	}

	for (j = 0; j < current->nx[1]; j++)
	{

		int idx = j * nrow;

		for (i = 0; i < current->nx[0]; i++)
		{

			// Get lower, central and upper values
			t_vfld fl = flbuf[i];
			t_vfld f0 = J[idx + i];
			t_vfld fu = J[idx + i + nrow];

			// Store the value that will be overritten for use in the next row
			flbuf[i] = f0;

			// Convolution with kernel
			t_vfld fs;
			fs.x = sa * fl.x + sb * f0.x + sa * fu.x;
			fs.y = sa * fl.y + sb * f0.y + sa * fu.y;
			fs.z = sa * fl.z + sb * f0.z + sa * fu.z;

			// Store result
			J[idx + i] = fs;
		}
	}
}

// Apply a binomial filter to reduce noise (X direction).
// Or, apply a compensation filter (if applicable)
void current_smooth_x(t_current *current, enum smooth_type type)
{
	// filter kernel [sa, sb, sa]
	t_fld sa, sb;

	switch (type)
	{
		case BINOMIAL:
			kernel_x(current, 0.25, 0.5);
			break;
		case COMPENSATED:
			get_smooth_comp(current->smooth.xlevel, &sa, &sb);
			kernel_x(current, sa, sb);
			break;
		default:
			break;
	}
}

// Apply a binomial filter to reduce noise (Y direction).
// Or, apply a compensation filter (if applicable)
void current_smooth_y(t_current *current, enum smooth_type type)
{
	// filter kernel [sa, sb, sa]
	t_fld sa, sb;

	switch (type)
	{
		case BINOMIAL:
			kernel_y(current, 0.25, 0.5);
			break;
		case COMPENSATED:
			get_smooth_comp(current->smooth.ylevel, &sa, &sb);
			kernel_y(current, sa, sb);
			break;
		default:
			break;
	}
}

/*********************************************************************************************
 Diagnostics
 *********************************************************************************************/

// Recreate a global buffer for a given direction
void current_reconstruct_global_buffer(t_current *current, float *global_buffer, const int offset_y,
                                     const int offset_x, const int sim_nrow, const int jc)
{
	t_vfld *restrict f = current->J;
	float *restrict p = global_buffer + offset_x + offset_y * sim_nrow;

	switch (jc)
	{
		case 0:
			for (int j = 0; j < current->nx[1]; j++)
			{
				for (int i = 0; i < current->nx[0]; i++)
				{
					p[i] = f[i].x;
				}
				p += sim_nrow;
				f += current->nrow;
			}
			break;
		case 1:
			for (int j = 0; j < current->nx[1]; j++)
			{
				for (int i = 0; i < current->nx[0]; i++)
				{
					p[i] = f[i].y;
				}
				p += sim_nrow;
				f += current->nrow;
			}
			break;
		case 2:
			for (int j = 0; j < current->nx[1]; j++)
			{
				for (int i = 0; i < current->nx[0]; i++)
				{
					p[i] = f[i].z;
				}
				p += sim_nrow;
				f += current->nrow;
			}
			break;
	}
}

// Save the reconstructed global buffer in the ZDF file format
void current_report(const float *restrict global_buffer, const int iter_num, const int true_nx[2],
                    const float box[2], const float dt, const char jc, const char path[128])
{
	char vfname[3] = "";

	// Pack the information
	vfname[0] = 'J';

	switch (jc)
	{
		case 0:
			vfname[1] = '1';
			break;
		case 1:
			vfname[1] = '2';
			break;
		case 2:
			vfname[1] = '3';
			break;
	}
	vfname[2] = 0;

	t_zdf_grid_axis axis[2];
	axis[0] = (t_zdf_grid_axis) {.min = 0.0, .max = box[0], .label = "x_1", .units = "c/\\omega_p"};

	axis[1] = (t_zdf_grid_axis) {.min = 0.0, .max = box[1], .label = "x_2", .units = "c/\\omega_p"};

	t_zdf_grid_info info = {.ndims = 2, .label = vfname, .units = "e \\omega_p^2 / c", .axis = axis};

	info.nx[0] = true_nx[0];
	info.nx[1] = true_nx[1];

	t_zdf_iteration iter = {.n = iter_num, .t = iter_num * dt, .time_units = "1/\\omega_p"};

	zdf_save_grid(global_buffer, &info, &iter, path);
}
