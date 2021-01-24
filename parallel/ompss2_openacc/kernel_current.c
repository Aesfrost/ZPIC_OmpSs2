/*********************************************************************************************
 ZPIC
 kernel_current.c

 Created by Nicolas Guidotti on 14/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#include "current.h"
#include "utilities.h"

#define LOCAL_BUFFER_SIZE 1024
#define MIN_VALUE(x, y) x < y ? x : y

// Set the current buffer to zero
void current_zero_openacc(t_current *current)
{
	// zero fields
	const int size = current->total_size;

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

// Apply multiple passes of a binomial filter to reduce noise (X direction).
// Then, pass a compensation filter (if applicable). OpenAcc Task
void current_smooth_x_openacc(t_current *current)
{
	const int size = current->total_size;
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;

	t_fld sa = 0.25;
	t_fld sb = 0.5;

	// binomial filter
	for (int i = 0; i < current->smooth.xlevel; i++)
	{
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
