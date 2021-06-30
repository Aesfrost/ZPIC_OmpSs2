/*********************************************************************************************
 ZPIC
 current.c

 Created by Ricardo Fonseca on 12/8/10.
 Modified by Nicolas Guidotti on 14/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#include "current.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <cuda.h>

#include "utilities.h"
#include "zdf.h"

/*********************************************************************************************
 Constructor / Destructor
 *********************************************************************************************/
void current_new(t_current *current, int nx[], t_fld box[], float dt, const int device)
{
	// Number of guard cells for linear interpolation
	int gc[2][2] = {{1, 2}, {1, 2}};

	// Allocate global array
	size_t size = (gc[0][0] + nx[0] + gc[0][1]) * (gc[1][0] + nx[1] + gc[1][1]);
	current->total_size = size;
	current->overlap_size = (gc[0][0] + nx[0] + gc[0][1]) * (gc[1][0] + gc[1][1]);

	current->J_buf = alloc_device_buffer(size * sizeof(t_vfld), device);

	assert(current->J_buf);
	memset(current->J_buf, 0, size * sizeof(t_vfld));

	// store nx and gc values
	for (int i = 0; i < 2; i++)
	{
		current->nx[i] = nx[i];
		current->gc[i][0] = gc[i][0];
		current->gc[i][1] = gc[i][1];
	}
	current->nrow = gc[0][0] + nx[0] + gc[0][1];

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

	current->moving_window = 0;
}

void current_delete(t_current *current)
{
	free_device_buffer(current->J_buf);
	current->J_buf = NULL;
}

// Set the overlap zone between adjacent regions (only the below zone)
void current_overlap_zone(t_current *current, t_current *current_below, const int device)
{
	current->J_below = current_below->J + (current_below->nx[1] - current_below->gc[1][0]) *
			current_below->nrow;

#ifdef ENABLE_ADVISE
	cuMemAdvise(current->J_below - current->gc[0][0], current->overlap_size, CU_MEM_ADVISE_SET_ACCESSED_BY, device);
#endif
}

// Set the current buffer to zero (OpenACC)
void current_zero_openacc(t_current *current)
{
	current->iter++;

	// zero fields
	#pragma acc parallel loop
	for(int i = 0; i < current->total_size; i++)
	{
		current->J_buf[i].x = 0.0f;
		current->J_buf[i].y = 0.0f;
		current->J_buf[i].z = 0.0f;
	}
}

/*********************************************************************************************
 Communication
 *********************************************************************************************/

// Each region is only responsible to do the reduction operation (y direction) in its bottom edge (OpenAcc)
void current_reduction_y_openacc(t_current *current)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;
	t_vfld *restrict const J_overlap = current->J_below;

	#pragma acc parallel loop independent collapse(2)
	for (int j = -current->gc[1][0]; j < current->gc[1][1]; j++)
	{
		for (int i = -current->gc[0][0]; i < current->nx[0] + current->gc[0][1]; i++)
		{
			J[i + j * nrow].x += J_overlap[i + (j + current->gc[1][0]) * nrow].x;
			J[i + j * nrow].y += J_overlap[i + (j + current->gc[1][0]) * nrow].y;
			J[i + j * nrow].z += J_overlap[i + (j + current->gc[1][0]) * nrow].z;

			J_overlap[i + (j + current->gc[1][0]) * nrow] = J[i + j * nrow];
		}
	}
}

// Current reduction between ghost cells in the x direction (OpenAcc)
void current_reduction_x_openacc(t_current *current)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;
	t_vfld *restrict const J_overlap = &current->J[current->nx[0]];

	#pragma acc parallel loop independent collapse(2)
	for (int j = -current->gc[1][0]; j < current->nx[1] + current->gc[1][1]; j++)
	{
		for (int i = -current->gc[0][0]; i < current->gc[0][1]; i++)
		{
			J[i + j * nrow].x += J_overlap[i + j * nrow].x;
			J[i + j * nrow].y += J_overlap[i + j * nrow].y;
			J[i + j * nrow].z += J_overlap[i + j * nrow].z;

			J_overlap[i + j * nrow] = J[i + j * nrow];
		}
	}
}

// Update the ghost cells in the y direction (only the bottom edge, OpenAcc)
void current_gc_update_y_openacc(t_current *current)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;
	t_vfld *restrict const J_overlap = current->J_below;

	#pragma acc parallel loop independent collapse(2)
	for (int i = -current->gc[0][0]; i < current->nx[0] + current->gc[0][1]; i++)
	{
		for (int j = -current->gc[1][0]; j < current->gc[1][1]; j++)
		{
			if(j < 0)
			{
				J[i + j * nrow] = J_overlap[i + (j + current->gc[1][0]) * nrow];
			}else
			{
				J_overlap[i + (j + current->gc[1][0]) * nrow] = J[i + j * nrow];
			}
		}
	}
}

/*********************************************************************************************
 Current Smoothing
 *********************************************************************************************/

// Apply multiple passes of a binomial filter to reduce noise (X direction).
// Then, pass a compensation filter (if applicable). OpenAcc Task
void current_smooth_x_openacc(t_current *current)
{
	const int size = current->total_size;
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;

	t_fld sa = 0.25;
	t_fld sb = 0.5;

	// Apply the binomial filter in the x direction
	for (int i = 0; i < current->smooth.xlevel; i++)
	{
		#pragma acc parallel loop gang vector_length(384)
		for (int j = -current->gc[1][0]; j < current->nx[1] + current->gc[1][1]; ++j)
		{
			t_vfld J_temp[LOCAL_BUFFER_SIZE + 2];
			#pragma acc cache(J_temp[0:LOCAL_BUFFER_SIZE + 2])

			for (int begin_idx = 0; begin_idx < current->nx[0]; begin_idx += LOCAL_BUFFER_SIZE)
			{
				const int batch = MIN_VALUE(current->nx[0] - begin_idx, LOCAL_BUFFER_SIZE);

				if (begin_idx == 0) J_temp[0] = J[j * nrow - 1];
				else J_temp[0] = J_temp[LOCAL_BUFFER_SIZE];

				#pragma acc loop vector
				for (int i = 0; i <= batch; i++)
					J_temp[i + 1] = J[begin_idx + i + j * nrow];

				#pragma acc loop vector
				for (int i = 0; i < batch; i++)
				{
					J[begin_idx + i + j * nrow].x = J_temp[i].x * sa + J_temp[i + 1].x * sb
													+ J_temp[i + 2].x * sa;
					J[begin_idx + i + j * nrow].y = J_temp[i].y * sa + J_temp[i + 1].y * sb
													+ J_temp[i + 2].y * sa;
					J[begin_idx + i + j * nrow].z = J_temp[i].z * sa + J_temp[i + 1].z * sb
													+ J_temp[i + 2].z * sa;
				}
			}

			if (!current->moving_window)
			{
				#pragma acc loop vector
				for (int i = -current->gc[0][0]; i < current->gc[0][1]; i++)
					if (i < 0) J[i + j * nrow] = J[current->nx[0] + i + j * nrow];
					else J[current->nx[0] + i + j * nrow] = J[i + j * nrow];
			}
		}
	}

	// Compensator
	if (current->smooth.xtype == COMPENSATED)
	{
		// Calculate the value of the compensator kernel for an n pass binomial kernel
		t_fld a, b, total;

		a = -1;
		b = (4.0 + 2.0 * current->smooth.xlevel) / current->smooth.xlevel;
		total = 2 * a + b;

		sa = a / total;
		sb = b / total;

		// Apply the filter in the x direction
		#pragma acc parallel loop gang vector_length(384)
		for (int j = -current->gc[1][0]; j < current->nx[1] + current->gc[1][1]; ++j)
		{
			t_vfld J_temp[LOCAL_BUFFER_SIZE + 2];
			#pragma acc cache(J_temp[0:LOCAL_BUFFER_SIZE + 2])

			for(int begin_idx = 0; begin_idx < current->nx[0]; begin_idx += LOCAL_BUFFER_SIZE)
			{
				const int batch = MIN_VALUE(current->nx[0] - begin_idx, LOCAL_BUFFER_SIZE);

				if(begin_idx == 0) J_temp[0] = J[j * nrow - 1];
				else J_temp[0] = J_temp[LOCAL_BUFFER_SIZE];

				#pragma acc loop vector
				for(int i = 0; i <= batch; i++)
					J_temp[i + 1] = J[begin_idx + i + j * nrow];

				#pragma acc loop vector
				for (int i = 0; i < batch; i++)
				{
					J[begin_idx + i + j * nrow].x = J_temp[i].x * sa + J_temp[i + 1].x * sb
							+ J_temp[i + 2].x * sa;
					J[begin_idx + i + j * nrow].y = J_temp[i].y * sa + J_temp[i + 1].y * sb
							+ J_temp[i + 2].y * sa;
					J[begin_idx + i + j * nrow].z = J_temp[i].z * sa + J_temp[i + 1].z * sb
							+ J_temp[i + 2].z * sa;
				}
			}

			if(!current->moving_window)
			{
				#pragma acc loop vector
				for (int i = -current->gc[0][0]; i < current->gc[0][1]; i++)
					if(i < 0) J[i + j * nrow] = J[current->nx[0] + i + j * nrow];
					else J[current->nx[0] + i + j * nrow] = J[i + j * nrow];
			}
		}
	}
}

// Apply the filter in the y direction (CPU)
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

			// Get lower, central and below values
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

// Apply a binomial filter to reduce noise (Y direction).
// Or, apply a compensation filter (if applicable). CPU task
void current_smooth_y(t_current *current, enum smooth_type type)
{
	t_fld a, b, total;

	switch (type)
	{
		case BINOMIAL:
			kernel_y(current, 0.25, 0.5);
			break;
		case COMPENSATED:
			// Calculate the value of the compensator kernel for an n pass binomial kernel
			a = -1;
			b = (4.0 + 2.0 * current->smooth.xlevel) / current->smooth.xlevel;
			total = 2 * a + b;
			kernel_y(current, a / total, b / total);
			break;
		default:
			break;
	}
}

/*********************************************************************************************
 Diagnostics
 *********************************************************************************************/

// Reconstruct the simulation grid from all the regions (electric current for a given coordinate)
void current_reconstruct_global_buffer(t_current *current, float *global_buffer, const int offset,
		const int jc)
{
	t_vfld *restrict f = current->J;
	float *restrict p = global_buffer + offset * current->nx[0];

	switch (jc)
	{
		case 0:
			for (int j = 0; j < current->nx[1]; j++)
			{
				for (int i = 0; i < current->nx[0]; i++)
				{
					p[i] = f[i].x;
				}
				p += current->nx[0];
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
				p += current->nx[0];
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
				p += current->nx[0];
				f += current->nrow;
			}
			break;
	}
}

// Save the reconstructed simulation grid (electric current) in the ZDF file format
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
	axis[0] = (t_zdf_grid_axis ) { .min = 0.0, .max = box[0], .label = "x_1", .units = "c/\\omega_p" };

	axis[1] = (t_zdf_grid_axis ) { .min = 0.0, .max = box[1], .label = "x_2", .units = "c/\\omega_p" };

	t_zdf_grid_info info = { .ndims = 2, .label = vfname, .units = "e \\omega_p^2 / c", .axis = axis };

	info.nx[0] = true_nx[0];
	info.nx[1] = true_nx[1];

	t_zdf_iteration iter = { .n = iter_num, .t = iter_num * dt, .time_units = "1/\\omega_p" };

	zdf_save_grid(global_buffer, &info, &iter, path);

}
