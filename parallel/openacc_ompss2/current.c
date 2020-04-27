/*
 *  current.c
 *  zpic
 *
 *  Created by Ricardo Fonseca on 12/8/10.
 *  Copyright 2010 Centro de Física dos Plasmas. All rights reserved.
 *
 */

#include "current.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "zdf.h"

void current_new(t_current *current, int nx[], t_fld box[], float dt)
{
	int i;

	// Number of guard cells for linear interpolation
	int gc[2][2] = {{1, 2}, {1, 2}};

	// Allocate global array
	size_t size;

	size = (gc[0][0] + nx[0] + gc[0][1]) * (gc[1][0] + nx[1] + gc[1][1]);
	current->total_size = size;
	current->overlap_zone = (gc[0][0] + nx[0] + gc[0][1]) * (gc[1][0] + gc[1][1]);

	current->J_buf = malloc(size * sizeof(t_vfld));
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
	current->smooth = (t_smooth) {.xtype = NONE, .ytype = NONE, .xlevel = 0, .ylevel = 0};

	// Initialize time information
	current->iter = 0;
	current->dt = dt;

	current->moving_window = 0;

	// Zero initial current
	// This is only relevant for diagnostics, current is always zeroed before deposition
	//current_zero(current);
}

void current_delete(t_current *current)
{
	free(current->J_buf);
	current->J_buf = NULL;
}

void current_zero(t_current *current)
{
	// zero fields
	size_t size;
	size = (current->gc[0][0] + current->nx[0] + current->gc[0][1])
			* (current->gc[1][0] + current->nx[1] + current->gc[1][1]) * sizeof(t_vfld);
	memset(current->J_buf, 0, size);

}

void current_overlap_zone(t_current *current, t_current *upper_current)
{
	current->J_upper = upper_current->J + (upper_current->nx[1] - upper_current->gc[1][0]) *
			upper_current->nrow;
}

// Each region is only responsible to do the reduction operation in its top edge
//
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

void current_reduction_x(t_current *current)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;
	t_vfld *restrict const J_overlap = &current->J[current->nx[0]];

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

	current->iter++;
}

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

void current_report(const t_current *current, const char jc)
{
	t_vfld *f;
	float *buf, *p;
	int i, j;
	char vfname[3];

	// Pack the information
	buf = malloc(current->nx[0] * current->nx[1] * sizeof(float));
	p = buf;
	f = current->J;
	vfname[0] = 'J';

	switch (jc)
	{
		case 0:
			for (j = 0; j < current->nx[1]; j++)
			{
				for (i = 0; i < current->nx[0]; i++)
				{
					p[i] = f[i].x;
				}
				p += current->nx[0];
				f += current->nrow;
			}
			vfname[1] = '1';
			break;
		case 1:
			for (j = 0; j < current->nx[1]; j++)
			{
				for (i = 0; i < current->nx[0]; i++)
				{
					p[i] = f[i].y;
				}
				p += current->nx[0];
				f += current->nrow;
			}
			vfname[1] = '2';
			break;
		case 2:
			for (j = 0; j < current->nx[1]; j++)
			{
				for (i = 0; i < current->nx[0]; i++)
				{
					p[i] = f[i].z;
				}
				p += current->nx[0];
				f += current->nrow;
			}
			vfname[1] = '3';
			break;
	}
	vfname[2] = 0;

	t_zdf_grid_axis axis[2];
	axis[0] = (t_zdf_grid_axis) {.min = 0.0, .max = current->box[0], .label = "x_1",
									.units = "c/\\omega_p"};

	axis[1] = (t_zdf_grid_axis) {.min = 0.0, .max = current->box[1], .label = "x_2",
									.units = "c/\\omega_p"};

	t_zdf_grid_info info = {.ndims = 2, .label = vfname, .units = "e \\omega_p^2 / c", .axis = axis};

	info.nx[0] = current->nx[0];
	info.nx[1] = current->nx[1];

	t_zdf_iteration iter = {.n = current->iter, .t = current->iter * current->dt,
							.time_units = "1/\\omega_p"};

	zdf_save_grid(buf, &info, &iter, "CURRENT");

	// free local data
	free(buf);

}

/*
 * get_smooth_comp
 *  Gets the value of the compensator kernel for an n pass binomial kernel
 */

//void get_smooth_comp(int n, t_fld *sa, t_fld *sb)
//{
//	t_fld a, b, total;
//
//	a = -1;
//	b = (4.0 + 2.0 * n) / n;
//	total = 2 * a + b;
//
//	*sa = a / total;
//	*sb = b / total;
//}

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

	// Update y boundaries

	// Grid is always periodic along y
	for (i = -current->gc[0][0]; i < current->nx[0] + current->gc[0][1]; i++)
	{
		for (j = -current->gc[1][0]; j < 0; j++)
			J[i + j * nrow] = J[i + (current->nx[1] + j) * nrow];
		for (j = 0; j < current->gc[1][1]; j++)
			J[i + (current->nx[1] + j) * nrow] = J[i + j * nrow];
	}

}

void current_smooth_x(t_current *current)
{
	// binomial filter
	for (int i = 0; i < current->smooth.xlevel; i++)
		kernel_x(current, 0.25, 0.5);

	// Compensator
	if (current->smooth.xtype == COMPENSATED)
	{
		t_fld a, b, total;

		a = -1;
		b = (4.0 + 2.0 * current->smooth.xlevel) / current->smooth.xlevel;
		total = 2 * a + b;

		kernel_x(current, a / total, b / total);
	}
}

//void current_smooth_y(t_current *current)
//{
//	// filter kernel [sa, sb, sa]
//	t_fld sa, sb;
//
//	// y-direction filtering
//	if (current->smooth.ytype != NONE)
//	{
//		// binomial filter
//		sa = 0.25;
//		sb = 0.5;
//		for (int i = 0; i < current->smooth.xlevel; i++)
//			kernel_y(current, 0.25, 0.5);
//
//
//		// Compensator
//		if (current->smooth.ytype == COMPENSATED)
//		{
//			get_smooth_comp(current->smooth.ylevel, &sa, &sb);
//			kernel_y(current, sa, sb);
//		}
//	}
//
//}

void kernel_x_openacc(t_current *const current, const t_fld sa, const t_fld sb)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;
	t_vfld *restrict J_buf = malloc(nrow * current->nx[1] * sizeof(t_vfld));
	t_vfld *restrict J_aux = J_buf + current->gc[0][0];

	#pragma acc parallel loop independent collapse(2)
	for (int j = 0; j < current->nx[1]; ++j)
		for (int i = 0; i < nrow; ++i)
			J_buf[i + j * nrow] = J[i - 1 + j * nrow];

	#pragma acc parallel loop independent collapse(2)
	for (int j = 0; j < current->nx[1]; ++j)
	{
		for (int i = 0; i < current->nx[0]; ++i)
		{
			J[i + j * nrow].x = J_aux[i - 1 + j * nrow].x * sa + J_aux[i + j * nrow].x * sb
					+ J_aux[i + 1 + j * nrow].x * sa;
			J[i + j * nrow].y = J_aux[i - 1 + j * nrow].y * sa + J_aux[i + j * nrow].y * sb
					+ J_aux[i + 1 + j * nrow].y * sa;
			J[i + j * nrow].z = J_aux[i - 1 + j * nrow].z * sa + J_aux[i + j * nrow].z * sb
					+ J_aux[i + 1 + j * nrow].z * sa;
		}
	}

	if(!current->moving_window)
	{
		#pragma acc parallel loop independent collapse(2)
		for (int j = 0; j < current->nx[1]; ++j)
		{
			for (int i = -current->gc[0][0]; i < current->gc[0][1]; i++)
				if(i < 0) J[i + j * nrow] = J[current->nx[0] + i + j * nrow];
				else J[current->nx[0] + i + j * nrow] = J[i + j * nrow];
		}
	}

	free(J_buf);
}

void current_smooth_x_openacc(t_current *current)
{
	// binomial filter
	for (int i = 0; i < current->smooth.xlevel; i++)
		kernel_x_openacc(current, 0.25, 0.5);

	// Compensator
	if (current->smooth.xtype == COMPENSATED)
	{
		t_fld a, b, total;

		a = -1;
		b = (4.0 + 2.0 * current->smooth.xlevel) / current->smooth.xlevel;
		total = 2 * a + b;

		kernel_x_openacc(current, a / total, b / total);
	}
}
