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

#include "zdf.h"

/*********************************************************************************************
 Constructor / Destructor
 *********************************************************************************************/
void current_new(t_current *current, int nx[], t_fld box[], float dt)
{
	int i;

	// Number of guard cells for linear interpolation
	int gc[2][2] = { { 1, 2 }, { 1, 2 } };

	// Allocate global array
	size_t size;

	size = (gc[0][0] + nx[0] + gc[0][1]) * (gc[1][0] + nx[1] + gc[1][1]);
	current->total_size = size;
	current->overlap_zone = (gc[0][0] + nx[0] + gc[0][1]) * (gc[1][0] + gc[1][1]);

	current->J_buf = calloc(size, sizeof(t_vfld));
	assert(current->J_buf);

	// store nx and gc values
	for (i = 0; i < 2; i++)
	{
		current->nx[i] = nx[i];
		current->gc[i][0] = gc[i][0];
		current->gc[i][1] = gc[i][1];
	}
	current->nrow = gc[0][0] + nx[0] + gc[0][1];

	// Make J point to cell [0][0]
	current->J = current->J_buf + gc[0][0] + gc[1][0] * current->nrow;

	// Set cell sizes and box limits
	for (i = 0; i < 2; i++)
	{
		current->box[i] = box[i];
		current->dx[i] = box[i] / nx[i];
	}

	// Clear smoothing options
	current->smooth = (t_smooth ) { .xtype = NONE, .ytype = NONE, .xlevel = 0, .ylevel = 0 };

	// Initialize time information
	current->iter = 0;
	current->dt = dt;

	current->moving_window = 0;
}

void current_delete(t_current *current)
{
	free(current->J_buf);
	current->J_buf = NULL;
}

// Set the current buffer to zero
void current_zero(t_current *current, const int start, const int size)
{
	memset(current->J_buf + start, 0, size * sizeof(t_vfld));
}

// Set the overlap zone between adjacent regions (only the upper zone)
void current_overlap_zone(t_current *current, t_current *upper_current)
{
	current->J_upper = upper_current->J
			+ (upper_current->nx[1] - upper_current->gc[1][0]) * upper_current->nrow;
}

/*********************************************************************************************
 Communication
 *********************************************************************************************/

// Each region is only responsible to do the reduction operation in its top edge
void current_reduction_y(t_current *current)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;
	t_vfld *restrict const J_overlap = current->J_upper;

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

// Current reduction between ghost cells in the x direction
void current_reduction_x(t_current *current, const int begin, const int end)
{
	if (current->moving_window) return;

	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;
	t_vfld *restrict const J_overlap = &current->J[current->nx[0]];

	for (int j = begin; j < end; j++)
	{
		for (int i = -current->gc[0][0]; i < current->gc[0][1]; i++)
		{
			J[i + j * nrow].x += J_overlap[i + j * nrow].x;
			J[i + j * nrow].y += J_overlap[i + j * nrow].y;
			J[i + j * nrow].z += J_overlap[i + j * nrow].z;

			J_overlap[i + j * nrow] = J[i + j * nrow];
		}
	}

	current->iter++;
}

// Update the ghost cells in the y direction (only the upper zone)
void current_gc_update_y(t_current *current)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;
	t_vfld *restrict const J_overlap = current->J_upper;

	for (int j = -current->gc[1][0]; j < 0; j++)
	{
		for (int i = -current->gc[0][0]; i < current->nx[0] + current->gc[0][1]; i++)
		{
			J[i + j * nrow] = J_overlap[i + (j + current->gc[1][0]) * nrow];
		}
	}

	for (int j = 0; j < current->gc[1][1]; j++)
	{
		for (int i = -current->gc[0][0]; i < current->nx[0] + current->gc[0][1]; i++)
		{
			J_overlap[i + (j + current->gc[1][0]) * nrow] = J[i + j * nrow];
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
	int i, j;
	t_vfld *restrict const J = current->J;
	const int nrow = current->nrow;

	for (j = 0; j < current->nx[1]; j++)
	{
		int idx = j * nrow;

		t_vfld fl = J[idx - 1];
		t_vfld f0 = J[idx];

		for (i = 0; i < current->nx[0]; i++)
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

		// Update x boundaries unless we are using a moving window
		if (!current->moving_window)
		{
			for (i = -current->gc[0][0]; i < 0; i++)
				J[idx + i] = J[idx + current->nx[0] + i];

			for (i = 0; i < current->gc[0][1]; i++)
				J[idx + current->nx[0] + i] = J[idx + i];
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

// Apply multiple passes of a binomial filter to reduce noise (X direction).
// Then, pass a compensation filter (if applicable)
void current_smooth_x(t_current *current)
{
	// filter kernel [sa, sb, sa]
	t_fld sa, sb;

	// binomial filter
	for (int i = 0; i < current->smooth.xlevel; i++)
		kernel_x(current, 0.25, 0.5);

	// Compensator
	if (current->smooth.xtype == COMPENSATED)
	{
		get_smooth_comp(current->smooth.xlevel, &sa, &sb);
		kernel_x(current, sa, sb);
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
	axis[0] = (t_zdf_grid_axis ) { .min = 0.0, .max = box[0], .label = "x_1", .units = "c/\\omega_p" };

	axis[1] = (t_zdf_grid_axis ) { .min = 0.0, .max = box[1], .label = "x_2", .units = "c/\\omega_p" };

	t_zdf_grid_info info = { .ndims = 2, .label = vfname, .units = "e \\omega_p^2 / c", .axis = axis };

	info.nx[0] = true_nx[0];
	info.nx[1] = true_nx[1];

	t_zdf_iteration iter = { .n = iter_num, .t = iter_num * dt, .time_units = "1/\\omega_p" };

	zdf_save_grid(global_buffer, &info, &iter, path);

}
