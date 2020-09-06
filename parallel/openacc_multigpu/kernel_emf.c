/*********************************************************************************************
 ZPIC
 kernel_emf.c

 Created by Nicolas Guidotti on 14/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/


#include "emf.h"
#include <stdlib.h>
#include <string.h>

#ifdef ENABLE_PREFETCH
#include <cuda.h>

void emf_prefetch_openacc(t_vfld *buf, const size_t size, const int device)
{
	cudaMemPrefetchAsync(buf, size * sizeof(t_vfld), device, NULL);
}
#endif

void yee_b_openacc(t_emf *emf, const float dt)
{
	// these must not be unsigned because we access negative cell indexes
	t_fld dt_dx, dt_dy;
	t_vfld *const restrict B = emf->B;
	const t_vfld *const restrict E = emf->E;
	const int nrow = emf->nrow;

	dt_dx = dt / emf->dx[0];
	dt_dy = dt / emf->dx[1];

	// Canonical implementation
	#pragma acc parallel loop independent tile(4, 4)
	for (int j = -1; j <= emf->nx[1]; j++)
	{
		for (int i = -1; i <= emf->nx[0]; i++)
		{
			B[i + j * nrow].x += (-dt_dy * (E[i + (j + 1) * nrow].z - E[i + j * nrow].z));
			B[i + j * nrow].y += (dt_dx * (E[(i + 1) + j * nrow].z - E[i + j * nrow].z));
			B[i + j * nrow].z += (-dt_dx * (E[(i + 1) + j * nrow].y - E[i + j * nrow].y)
					+ dt_dy * (E[i + (j + 1) * nrow].x - E[i + j * nrow].x));
		}
	}
}

void yee_e_openacc(t_emf *emf, const t_current *current, const float dt)
{
	// these must not be unsigned because we access negative cell indexes
	const int nrow_e = emf->nrow;
	const int nrow_j = current->nrow;

	t_fld dt_dx, dt_dy;

	dt_dx = dt / emf->dx[0];
	dt_dy = dt / emf->dx[1];

	t_vfld *const restrict E = emf->E;
	const t_vfld *const restrict B = emf->B;
	const t_vfld *const restrict J = current->J;

	// Canonical implementation
	#pragma acc parallel loop independent tile(4, 4)
	for (int j = 0; j <= emf->nx[1] + 1; j++)
	{
		for (int i = 0; i <= emf->nx[0] + 1; i++)
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
void emf_gc_x_openacc(t_emf *emf)
{
	// For moving window don't update x boundaries
	if (!emf->moving_window)
	{
		const int nrow = emf->nrow;

		t_vfld *const restrict E = emf->E;
		t_vfld *const restrict B = emf->B;

		// x
		#pragma acc parallel loop collapse(2) independent
		for (int j = -emf->gc[1][0]; j < emf->nx[1] + emf->gc[1][1]; j++)
		{
			// lower
			for (int i = -emf->gc[0][0]; i < emf->gc[0][1]; i++)
			{
				if(i < 0)
				{
					E[i + j * nrow].x = E[emf->nx[0] + i + j * nrow].x;
					E[i + j * nrow].y = E[emf->nx[0] + i + j * nrow].y;
					E[i + j * nrow].z = E[emf->nx[0] + i + j * nrow].z;

					B[i + j * nrow].x = B[emf->nx[0] + i + j * nrow].x;
					B[i + j * nrow].y = B[emf->nx[0] + i + j * nrow].y;
					B[i + j * nrow].z = B[emf->nx[0] + i + j * nrow].z;
				}else
				{
					E[emf->nx[0] + i + j * nrow].x = E[i + j * nrow].x;
					E[emf->nx[0] + i + j * nrow].y = E[i + j * nrow].y;
					E[emf->nx[0] + i + j * nrow].z = E[i + j * nrow].z;

					B[emf->nx[0] + i + j * nrow].x = B[i + j * nrow].x;
					B[emf->nx[0] + i + j * nrow].y = B[i + j * nrow].y;
					B[emf->nx[0] + i + j * nrow].z = B[i + j * nrow].z;
				}
			}
		}
	}
}

// Update ghost cells in the upper overlap zone (Y direction, OpenAcc)
void emf_update_gc_y_openacc(t_emf *emf, const int device)
{
	const int nrow = emf->nrow;

	t_vfld *const restrict E = emf->E;
	t_vfld *const restrict B = emf->B;
	t_vfld *const restrict E_overlap = emf->E_upper;
	t_vfld *const restrict B_overlap = emf->B_upper;

#ifdef ENABLE_PREFETCH
	const int size_overlap = emf->overlap_size;
	const int size = emf->total_size;
	emf_prefetch_openacc(emf->B_buf, size, device);
	emf_prefetch_openacc(emf->E_buf, size, device);
	emf_prefetch_openacc(E_overlap, size_overlap, device);
	emf_prefetch_openacc(B_overlap, size_overlap, device);
#endif

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
void emf_move_window_openacc(t_emf *emf, const int device)
{
	if ((emf->iter * emf->dt) > emf->dx[0] * (emf->n_move + 1))
	{
		const int nrow = emf->nrow;
		size_t size = emf->total_size;
		t_vfld *const restrict E = malloc(size * sizeof(t_vfld));
		t_vfld *const restrict B = malloc(size * sizeof(t_vfld));

		const t_vfld zero_fld = {0., 0., 0.};

		// Increase moving window counter
		emf->n_move++;

#ifdef ENABLE_PREFETCH
	emf_prefetch_openacc(B, size, device);
	emf_prefetch_openacc(E, size, device);
#endif

		#pragma acc parallel loop
		for(int i = 0; i < size; i++)
		{
			E[i] = emf->E_buf[i];
			B[i] = emf->B_buf[i];
		}

		// Shift data left 1 cell and zero rightmost cells
		#pragma acc parallel loop independent collapse(2)
		for (int j = 0; j < emf->gc[1][0] + emf->nx[1] + emf->gc[1][1]; j++)
		{
			for (int i = 0; i < nrow; i++)
			{
				if (i < emf->gc[0][0] + emf->nx[0] - 1)
				{
					emf->E_buf[i + j * nrow] = E[i + j * nrow + 1];
					emf->B_buf[i + j * nrow] = B[i + j * nrow + 1];
				} else
				{
					emf->E_buf[i + j * nrow] = zero_fld;
					emf->B_buf[i + j * nrow] = zero_fld;
				}
			}
		}

		free(E);
		free(B);
	}
}

// Perform the local integration of the fields (and post processing). OpenAcc Task
void emf_advance_openacc(t_emf *emf, const t_current *current, const int device)
{
	const float dt = emf->dt;

#ifdef ENABLE_PREFETCH
	emf_prefetch_openacc(emf->B_buf, emf->total_size, device);
	emf_prefetch_openacc(emf->E_buf, emf->total_size, device);
	current_prefetch_openacc(current->J_buf, current->total_size, device);
#endif

	// Advance EM field using Yee algorithm modified for having E and B time centered
	yee_b_openacc(emf, dt / 2.0f);
	yee_e_openacc(emf, current, dt);
	yee_b_openacc(emf, dt / 2.0f);

	// Update guard cells with new values
	emf_gc_x_openacc(emf);

	// Advance internal iteration number
	emf->iter += 1;

	// Move simulation window if needed
	if (emf->moving_window) emf_move_window_openacc(emf, device);
}

