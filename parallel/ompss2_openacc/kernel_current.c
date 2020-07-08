/*********************************************************************************************
 ZPIC
 kernel_current.c

 Created by Nicolas Guidotti on 14/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#include "current.h"
#include "utilities.h"

// Set the current buffer to zero
void current_zero_openacc(t_current *current)
{
	// zero fields
	unsigned int size = (current->gc[0][0] + current->nx[0] + current->gc[0][1])
			* (current->gc[1][0] + current->nx[1] + current->gc[1][1]);

	#pragma acc parallel loop
	for(int i = 0; i < size; i++)
	{
		current->J_buf[i].x = 0.0f;
		current->J_buf[i].y = 0.0f;
		current->J_buf[i].z = 0.0f;
	}
}

// Each region is only responsible to do the reduction operation (y direction) in its top edge (OpenAcc)
void current_reduction_y_openacc(t_current *current)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;
	t_vfld *restrict const J_overlap = current->J_upper;

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

	current->iter++;
}

// Update the ghost cells in the y direction (only the upper zone, OpenAcc)
void current_gc_update_y_openacc(t_current *current)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;
	t_vfld *restrict const J_overlap = current->J_upper;

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

// Apply the filter in the x direction (OpenAcc)
void kernel_x_openacc(t_current *const current, const t_fld sa, const t_fld sb)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;
	t_vfld *restrict J_aux = current->J_temp + current->gc[0][0] + current->gc[1][0] * current->nrow;

	#pragma acc parallel loop gang
	for (int j = 0; j < current->nx[1]; ++j)
	{
		#pragma acc loop vector
		for(int i = -current->gc[0][0]; i < current->nx[0] + current->gc[0][0]; i++)
			J_aux[i + j * nrow] = J[i + j * nrow];

		#pragma acc loop vector
		for (int i = 0; i < current->nx[0]; ++i)
		{
			J[i + j * nrow].x = J_aux[i - 1 + j * nrow].x * sa + J_aux[i + j * nrow].x * sb
					+ J_aux[i + 1 + j * nrow].x * sa;
			J[i + j * nrow].y = J_aux[i - 1 + j * nrow].y * sa + J_aux[i + j * nrow].y * sb
					+ J_aux[i + 1 + j * nrow].y * sa;
			J[i + j * nrow].z = J_aux[i - 1 + j * nrow].z * sa + J_aux[i + j * nrow].z * sb
					+ J_aux[i + 1 + j * nrow].z * sa;
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

// Apply multiple passes of a binomial filter to reduce noise (X direction).
// Then, pass a compensation filter (if applicable). OpenAcc Task
void current_smooth_x_openacc(t_current *current)
{
	if (!current->J_temp) current->J_temp = alloc_align_buffer(DEFAULT_ALIGNMENT, current->total_size * sizeof(t_vfld));

	// binomial filter
	for (int i = 0; i < current->smooth.xlevel; i++)
		kernel_x_openacc(current, 0.25, 0.5);

	// Compensator
	if (current->smooth.xtype == COMPENSATED)
	{
		// Calculate the value of the compensator kernel for an n pass binomial kernel
		t_fld a, b, total;

		a = -1;
		b = (4.0 + 2.0 * current->smooth.xlevel) / current->smooth.xlevel;
		total = 2 * a + b;

		kernel_x_openacc(current, a / total, b / total);
	}
}
