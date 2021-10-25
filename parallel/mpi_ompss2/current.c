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
#include "task_management.h"

static MPI_Datatype MPI_VFLD = MPI_DATATYPE_NULL;

/*********************************************************************************************
 Constructor / Destructor
 *********************************************************************************************/
void current_new(t_current *current, int nx[], t_fld box[], float dt, bool on_right_edge,
                 bool on_left_edge)
{
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
	for (int i = 0; i < 2; i++)
	{
		current->nx[i] = nx[i];
		current->gc[i][0] = gc[i][0];
		current->gc[i][1] = gc[i][1];
	}

	// Make J point to cell [0][0]
	current->J = current->J_buf + gc[0][0] + gc[1][0] * current->nrow;

	// Set cell sizes and box limits
	for (int i = 0; i < 2; i++)
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

	current->on_left_edge = on_left_edge;
	current->on_right_edge = on_right_edge;

	if(MPI_VFLD == MPI_DATATYPE_NULL)
	{
		CHECK_MPI_ERROR(MPI_Type_contiguous(3, MPI_FLOAT, &MPI_VFLD));
		CHECK_MPI_ERROR(MPI_Type_commit(&MPI_VFLD));
	}

	// Reset all MPI requests
	current->num_mpi_requests = 0;
	for (int i = 0; i < 4; ++i)
		current->mpi_requests[i] = MPI_REQUEST_NULL;
}

void current_delete(t_current *current)
{
	free(current->J_buf);
	current->J_buf = NULL;

	for (int i = 0; i < NUM_ADJ_GRID; ++i)
	{
		if(current->inter_proc_comm[i])
		{
			free(current->send_J[i]);
			free(current->receive_J[i]);
		}
	}

	if(MPI_VFLD != MPI_DATATYPE_NULL)
	{
		CHECK_MPI_ERROR(MPI_Type_free(&MPI_VFLD));
		MPI_VFLD = MPI_DATATYPE_NULL;
	}
}

// Set the current buffer to zero
void current_zero(t_current *current)
{
	current->iter++;

	// zero fields
	size_t size = current->nrow * current->ncol;
	memset(current->J_buf, 0, size * sizeof(t_vfld));
}

/*********************************************************************************************
 Communication
 *********************************************************************************************/

void current_link_adj_regions(t_current *current, t_current *current_down, t_current *current_up)
{
	const int segm_nrow = current->gc[0][0] + current->gc[0][1];

	for (int dir = 0; dir < 4; dir++)
	{
		switch (dir)
		{
			case GRID_DOWN:
				if (current_down)   // The adjacent region (bottom) is in the same process
				{
					current->send_J[dir] = current_down->J_buf + current_down->nx[1] * current_down->nrow;
					current->receive_J[dir] = current->send_J[dir];
					current->inter_proc_comm[dir] = false;

				} else
				{
					current->send_J[dir] = calloc(current->overlap_size, sizeof(t_vfld));
					current->receive_J[dir] = calloc(current->overlap_size, sizeof(t_vfld));
					current->inter_proc_comm[dir] = true;
				}
				break;

			case GRID_UP:
				if (current_up)   // The adjacent region (upper) is in the same process
				{
					// Ghost cells in the left
					current->send_J[dir] = current_up->J_buf;
					current->receive_J[dir] = current->send_J[dir];
					current->inter_proc_comm[dir] = false;

				} else
				{
					current->send_J[dir] = calloc(current->overlap_size, sizeof(t_vfld));
					current->receive_J[dir] = calloc(current->overlap_size, sizeof(t_vfld));
					current->inter_proc_comm[dir] = true;
				}
				break;

			default:   // GRID_LEFT or GRID_RIGHT

				// Offset to the beginning of the region (remember that region limits are in global coordinates)
				current->send_J[dir] = calloc(current->ncol * segm_nrow, sizeof(t_vfld));
				current->receive_J[dir] = calloc(current->ncol * segm_nrow, sizeof(t_vfld));
				current->inter_proc_comm[dir] = true;
				break;
		}
	}
}

void current_exchange_gc_x(t_current *current, const int region_id, const unsigned int adj_ranks[4])
{
	const int segm_nrow = current->gc[0][0] + current->gc[0][1];
	const int nrow = current->nrow;

	t_vfld *restrict J = current->J_buf;
	t_vfld *restrict J_left = current->send_J[GRID_LEFT];
	t_vfld *restrict J_right = current->send_J[GRID_RIGHT];

	current->num_mpi_requests = 0;
	for (int i = 0; i < 4; ++i)
		current->mpi_requests[i] = MPI_REQUEST_NULL;

	if (!current->moving_window || !current->on_left_edge)
	{
		for (int j = 0; j < current->ncol; ++j)
			for (int i = 0; i < segm_nrow; ++i)
				J_left[i + j * segm_nrow] = J[i + j * nrow];

		CHECK_MPI_ERROR(MPI_Isend(current->send_J[GRID_LEFT],
		                          current->ncol * segm_nrow,
		                          MPI_VFLD,
		                          adj_ranks[GRID_LEFT],
		                          CREATE_MPI_TAG(GRID_RIGHT, region_id, MPI_TAG_J),
		                          MPI_COMM_WORLD,
		                          &current->mpi_requests[current->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Irecv(current->receive_J[GRID_LEFT],
		                          current->ncol * segm_nrow,
		                          MPI_VFLD,
		                          adj_ranks[GRID_LEFT],
		                          CREATE_MPI_TAG(GRID_LEFT, region_id, MPI_TAG_J),
		                          MPI_COMM_WORLD,
		                          &current->mpi_requests[current->num_mpi_requests++]));
	}

	if (!current->moving_window || !current->on_right_edge)
	{
		for (int j = 0; j < current->ncol; ++j)
			for (int i = 0; i < segm_nrow; ++i)
				J_right[i + j * segm_nrow] = J[current->nx[0] + i + j * nrow];

		CHECK_MPI_ERROR(MPI_Isend(current->send_J[GRID_RIGHT],
		                          current->ncol * segm_nrow,
		                          MPI_VFLD,
		                          adj_ranks[GRID_RIGHT],
		                          CREATE_MPI_TAG(GRID_LEFT, region_id, MPI_TAG_J),
		                          MPI_COMM_WORLD,
		                          &current->mpi_requests[current->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Irecv(current->receive_J[GRID_RIGHT],
		                          current->ncol * segm_nrow,
		                          MPI_VFLD,
		                          adj_ranks[GRID_RIGHT],
		                          CREATE_MPI_TAG(GRID_RIGHT, region_id, MPI_TAG_J),
		                          MPI_COMM_WORLD,
		                          &current->mpi_requests[current->num_mpi_requests++]));
	}
}

void current_reduction_x(t_current *current)
{
	const int nrow = current->nrow;
	const int segm_nrow = current->gc[0][0] + current->gc[0][1];

	t_vfld *restrict J = current->J_buf;
	t_vfld *restrict J_left = current->receive_J[GRID_LEFT];
	t_vfld *restrict J_right = current->receive_J[GRID_RIGHT];

	mpi_wait_async_comm(current->mpi_requests, current->num_mpi_requests);

	if (!current->moving_window || !current->on_left_edge)
	{
		for (int j = 0; j < current->ncol; ++j)
		{
			for (int i = 0; i < segm_nrow; ++i)
			{
				J[i + j * nrow].x += J_left[i + j * segm_nrow].x;
				J[i + j * nrow].y += J_left[i + j * segm_nrow].y;
				J[i + j * nrow].z += J_left[i + j * segm_nrow].z;
			}
		}
	}

	if (!current->moving_window || !current->on_right_edge)
	{
		for (int j = 0; j < current->ncol; ++j)
		{
			for (int i = 0; i < segm_nrow; ++i)
			{
				J[current->nx[0] + i + j * nrow].x += J_right[i + j * segm_nrow].x;
				J[current->nx[0] + i + j * nrow].y += J_right[i + j * segm_nrow].y;
				J[current->nx[0] + i + j * nrow].z += J_right[i + j * segm_nrow].z;
			}
		}
	}
}

void current_update_gc_x(t_current *current)
{
	const int nrow = current->nrow;
	const int segm_nrow = current->gc[0][0] + current->gc[0][1];

	t_vfld *restrict J = current->J_buf;
	t_vfld *restrict J_left = current->receive_J[GRID_LEFT];
	t_vfld *restrict J_right = current->receive_J[GRID_RIGHT];

	mpi_wait_async_comm(current->mpi_requests, current->num_mpi_requests);

	if (!(current->moving_window && current->on_left_edge))
		for (int j = 0; j < current->ncol; ++j)
			for (int i = 0; i < current->gc[0][0]; ++i)
				J[i + j * nrow] = J_left[i + j * segm_nrow];

	if (!(current->moving_window && current->on_right_edge))
		for (int j = 0; j < current->ncol; ++j)
			for (int i = current->gc[0][0]; i < segm_nrow; ++i)
				J[current->nx[0] + i + j * nrow] = J_right[i + j * segm_nrow];
}


void current_exchange_gc_y(t_current *current, const unsigned int adj_ranks[4])
{
	current->num_mpi_requests = 0;
	for (int i = 0; i < 4; ++i)
		current->mpi_requests[i] = MPI_REQUEST_NULL;

	if (current->inter_proc_comm[GRID_DOWN])
	{
		memcpy(current->send_J[GRID_DOWN], current->J_buf, current->overlap_size * sizeof(t_vfld));

		CHECK_MPI_ERROR(MPI_Isend(current->send_J[GRID_DOWN],
		                          current->overlap_size,
		                          MPI_VFLD,
		                          adj_ranks[GRID_DOWN],
		                          CREATE_MPI_TAG(GRID_UP, 0, MPI_TAG_J),
		                          MPI_COMM_WORLD,
		                          &current->mpi_requests[current->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Irecv(current->receive_J[GRID_DOWN],
		                          current->overlap_size,
		                          MPI_VFLD,
		                          adj_ranks[GRID_DOWN],
		                          CREATE_MPI_TAG(GRID_DOWN, 0, MPI_TAG_J),
		                          MPI_COMM_WORLD,
		                          &current->mpi_requests[current->num_mpi_requests++]));
	}

	if (current->inter_proc_comm[GRID_UP])
	{
		memcpy(current->send_J[GRID_UP], current->J_buf + current->nx[1] * current->nrow,
		       current->overlap_size * sizeof(t_vfld));

		CHECK_MPI_ERROR(MPI_Isend(current->send_J[GRID_UP],
		                          current->overlap_size,
		                          MPI_VFLD,
		                          adj_ranks[GRID_UP],
		                          CREATE_MPI_TAG(GRID_DOWN, 0, MPI_TAG_J),
		                          MPI_COMM_WORLD,
		                          &current->mpi_requests[current->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Irecv(current->receive_J[GRID_UP],
		                          current->overlap_size,
		                          MPI_VFLD,
		                          adj_ranks[GRID_UP],
		                          CREATE_MPI_TAG(GRID_UP, 0, MPI_TAG_J),
		                          MPI_COMM_WORLD,
		                          &current->mpi_requests[current->num_mpi_requests++]));
	}
}

// Each region is only responsible to do the reduction operation in its bottom edge
void current_reduction_y(t_current *current)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J_buf;
	t_vfld *restrict const J_down = current->receive_J[GRID_DOWN];

	mpi_wait_async_comm(current->mpi_requests, current->num_mpi_requests);

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

	if (current->inter_proc_comm[GRID_UP])
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
