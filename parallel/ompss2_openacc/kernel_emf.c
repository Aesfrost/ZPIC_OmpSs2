/*********************************************************************************************
 ZPIC
 kernel_emf.c

 Created by Nicolas Guidotti on 14/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/
#include "emf.h"
#include <stdlib.h>
#include <string.h>

#define LOCAL_BUFFER_SIZE 1024

#pragma oss task device(openacc) inout(E[-gc[0][0] - gc[1][0] * nrow; total_size]) \
	inout(B[-gc[0][0] - gc[1][0] * nrow; total_size]) \
	label("EMF Advance (GPU, Yee B)")
void yee_b_openacc(t_vfld *restrict B, const t_vfld *restrict E, const t_fld dt_dx,
        const t_fld dt_dy, const int nrow, const int nx[2], const int gc[2][2], const int total_size)
{
	// Canonical implementation
	#pragma acc parallel loop independent tile(16, 16)
	for (int j = -1; j <= nx[1]; j++)
	{
		for (int i = -1; i <= nx[0]; i++)
		{
			B[i + j * nrow].x += (-dt_dy * (E[i + (j + 1) * nrow].z - E[i + j * nrow].z));
			B[i + j * nrow].y += (dt_dx * (E[(i + 1) + j * nrow].z - E[i + j * nrow].z));
			B[i + j * nrow].z += (-dt_dx * (E[(i + 1) + j * nrow].y - E[i + j * nrow].y)
					+ dt_dy * (E[i + (j + 1) * nrow].x - E[i + j * nrow].x));
		}
	}
}

#pragma oss task device(openacc) inout(E[-gc[0][0] - gc[1][0] * nrow_e; total_size]) \
	inout(B[-gc[0][0] - gc[1][0] * nrow_e; total_size]) \
	in(J[-gc[0][0] - gc[1][0] * nrow_j; total_size]) label("EMF Advance (GPU, Yee E)")
void yee_e_openacc(const t_vfld *restrict B, t_vfld *restrict E, const t_vfld *restrict J,
        const const t_fld dt_dx, const t_fld dt_dy, const float dt, const int nrow_e,
        const int nrow_j, const int nx[2], const int gc[2][2], const int total_size)
{
	// Canonical implementation
	#pragma acc parallel loop independent tile(16, 16)
	for (int j = 0; j <= nx[1] + 1; j++)
	{
		for (int i = 0; i <= nx[0] + 1; i++)
		{
			E[i + j * nrow_e].x += (+dt_dy * (B[i + j * nrow_e].z - B[i + (j - 1) * nrow_e].z))
					- dt * J[i + j * nrow_j].x;
			E[i + j * nrow_e].y += (-dt_dx * (B[i + j * nrow_e].z - B[(i - 1) + j * nrow_e].z))
					- dt * J[i + j * nrow_j].y;
			E[i + j * nrow_e].z += (+dt_dx * (B[i + j * nrow_e].y - B[(i - 1) + j * nrow_e].y)
					- dt_dy * (B[i + j * nrow_e].x - B[i + (j - 1) * nrow_e].x)) - dt * J[i + j * nrow_j].z;
		}
	}
}

// Update the ghost cells in the X direction (OpenAcc)
#pragma oss task device(openacc) inout(E[-gc[0][0] - gc[1][0] * nrow; total_size]) \
	inout(B[-gc[0][0] - gc[1][0] * nrow; total_size]) label("EMF Advance (GPU, GC X)")
void emf_update_gc_x_openacc(t_vfld *restrict E, t_vfld *restrict B, const int nrow, const int nx[2],
        const int gc[2][2], const int total_size, int *iter)
{
	#pragma acc parallel loop collapse(2) independent
	for (int j = -gc[1][0]; j < nx[1] + gc[1][1]; j++)
	{
		for (int i = -gc[0][0]; i < gc[0][1]; i++)
		{
			if (i < 0)
			{
				E[i + j * nrow].x = E[nx[0] + i + j * nrow].x;
				E[i + j * nrow].y = E[nx[0] + i + j * nrow].y;
				E[i + j * nrow].z = E[nx[0] + i + j * nrow].z;

				B[i + j * nrow].x = B[nx[0] + i + j * nrow].x;
				B[i + j * nrow].y = B[nx[0] + i + j * nrow].y;
				B[i + j * nrow].z = B[nx[0] + i + j * nrow].z;
			} else
			{
				E[nx[0] + i + j * nrow].x = E[i + j * nrow].x;
				E[nx[0] + i + j * nrow].y = E[i + j * nrow].y;
				E[nx[0] + i + j * nrow].z = E[i + j * nrow].z;

				B[nx[0] + i + j * nrow].x = B[i + j * nrow].x;
				B[nx[0] + i + j * nrow].y = B[i + j * nrow].y;
				B[nx[0] + i + j * nrow].z = B[i + j * nrow].z;
			}
		}
	}

	// Advance internal iteration number
	*iter += 1;
}

// Update ghost cells in the upper overlap zone (Y direction, OpenAcc)
void emf_update_gc_y_openacc(t_emf *emf)
{
	const int nrow = emf->nrow;

	t_vfld *const restrict E = emf->E;
	t_vfld *const restrict B = emf->B;
	t_vfld *const restrict E_overlap = emf->E_upper;
	t_vfld *const restrict B_overlap = emf->B_upper;

	// y
	#pragma acc parallel loop collapse(2) independent
	for (int i = -emf->gc[0][0]; i < emf->nx[0] + emf->gc[0][1]; i++)
	{
		for (int j = -emf->gc[1][0]; j < emf->gc[1][1]; j++)
		{
			if(j < 0)
			{
				B[i + j * nrow] = B_overlap[i + (j + emf->gc[1][0]) * nrow];
				E[i + j * nrow] = E_overlap[i + (j + emf->gc[1][0]) * nrow];
			}else
			{
				B_overlap[i + (j + emf->gc[1][0]) * nrow] = B[i + j * nrow];
				E_overlap[i + (j + emf->gc[1][0]) * nrow] = E[i + j * nrow];
			}
		}
	}
}

// Move the simulation window
#pragma oss task device(openacc) inout(E[0; total_size]) inout(B[0; total_size]) \
	label("EMF Advance (GPU, Move Window)")
void emf_move_window_openacc(t_vfld *restrict E, t_vfld *restrict B, int *n_move, const int nrow,
        const int gc[2][2], const int nx[2], const int total_size, const t_fld dt, const t_fld dx,
        int *iter)
{
	const t_vfld zero_fld = {0, 0, 0};

	// Advance internal iteration number
	*iter += 1;

	if ((*iter * dt) > dx * (*n_move + 1))
	{
		// Increase moving window counter
		(*n_move)++;

		// Shift data left 1 cell and zero rightmost cells
		#pragma acc parallel loop gang vector_length(384)
		for (int j = 0; j < gc[1][0] + nx[1] + gc[1][1]; j++)
		{
			t_vfld B_temp[LOCAL_BUFFER_SIZE];
			t_vfld E_temp[LOCAL_BUFFER_SIZE];

			#pragma acc cache(B_temp[0:LOCAL_BUFFER_SIZE])
			#pragma acc cache(E_temp[0:LOCAL_BUFFER_SIZE])

			for(int begin_idx = 0; begin_idx < nrow; begin_idx += LOCAL_BUFFER_SIZE)
			{
				#pragma acc loop vector
				for(int i = 0; i < LOCAL_BUFFER_SIZE; i++)
				{
					if((begin_idx + i) < gc[0][0] + nx[0] - 1)
					{
						B_temp[i] = B[begin_idx + 1 + i + j * nrow];
						E_temp[i] = E[begin_idx + 1 + i + j * nrow];
					}else
					{
						B_temp[i] = zero_fld;
						E_temp[i] = zero_fld;
					}
				}

				#pragma acc loop vector
				for(int i = 0; i < LOCAL_BUFFER_SIZE; i++)
				{
					if(begin_idx + i < nrow)
					{
						E[begin_idx + i + j * nrow] = E_temp[i];
						B[begin_idx + i + j * nrow] = B_temp[i];
					}
				}
			}
		}
	}
}

// Perform the local integration of the fields (and post processing). OpenAcc Task
void emf_advance_openacc(t_emf *emf, const t_current *current)
{
	const t_fld dt = emf->dt;
	const t_fld dt_dx = dt / emf->dx[0];
	const t_fld dt_dy = dt / emf->dx[1];

	// Advance EM field using Yee algorithm modified for having E and B time centered
	yee_b_openacc(emf->B, emf->E, dt_dx / 2.0f, dt_dy / 2.0f, emf->nrow, emf->nx, emf->gc, emf->total_size);
	yee_e_openacc(emf->B, emf->E, current->J, dt_dx, dt_dy, dt, emf->nrow, current->nrow, emf->nx, emf->gc,
	        emf->total_size);
	yee_b_openacc(emf->B, emf->E, dt_dx / 2.0f, dt_dy / 2.0f, emf->nrow, emf->nx, emf->gc, emf->total_size);

	if(emf->moving_window)
	{
		// Move simulation window
		emf_move_window_openacc(emf->E_buf, emf->B_buf, &emf->n_move, emf->nrow, emf->gc, emf->nx,
		        emf->total_size, dt, emf->dx[0], &emf->iter);
	} else
	{
		// Update guard cells with new values
		emf_update_gc_x_openacc(emf->E, emf->B, emf->nrow, emf->nx, emf->gc, emf->total_size,
		        &emf->iter);
	}
}

