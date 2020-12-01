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

#ifdef ENABLE_PREFETCH
#include <cuda.h>

void emf_prefetch_openacc(t_vfld *buf, const size_t size, const int device, void *stream)
{
	cudaMemPrefetchAsync(buf, size * sizeof(t_vfld), device, stream);
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
	#pragma acc parallel loop independent tile(16, 16)
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
	#pragma acc parallel loop independent tile(16, 16)
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
	void *stream = NULL;
	const int size_overlap = emf->overlap_size;
	emf_prefetch_openacc(E_overlap, size_overlap, device, stream);
	emf_prefetch_openacc(B_overlap, size_overlap, device, stream);
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
void emf_move_window_openacc(t_emf *emf)
{
	if ((emf->iter * emf->dt) > emf->dx[0] * (emf->n_move + 1))
	{
		const int nrow = emf->nrow;
		size_t size = emf->total_size;
		const t_vfld zero_fld = {0., 0., 0.};

		// Increase moving window counter
		emf->n_move++;

		// Shift data left 1 cell and zero rightmost cells
		#pragma acc parallel loop gang vector_length(384)
		for (int j = 0; j < emf->gc[1][0] + emf->nx[1] + emf->gc[1][1]; j++)
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
					if((begin_idx + i) < emf->gc[0][0] + emf->nx[0] - 1)
					{
						B_temp[i] = emf->B_buf[begin_idx + 1 + i + j * nrow];
						E_temp[i] = emf->E_buf[begin_idx + 1 + i + j * nrow];
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
						emf->E_buf[begin_idx + i + j * nrow] = E_temp[i];
						emf->B_buf[begin_idx + i + j * nrow] = B_temp[i];
					}
				}
			}
		}
	}
}

// Perform the local integration of the fields (and post processing). OpenAcc Task
void emf_advance_openacc(t_emf *emf, const t_current *current, const int device)
{
	const float dt = emf->dt;

	// Advance EM field using Yee algorithm modified for having E and B time centered
	yee_b_openacc(emf, dt / 2.0f);
	yee_e_openacc(emf, current, dt);
	yee_b_openacc(emf, dt / 2.0f);

	// Update guard cells with new values
	emf_gc_x_openacc(emf);

	// Advance internal iteration number
	emf->iter += 1;

	// Move simulation window if needed
	if (emf->moving_window) emf_move_window_openacc(emf);
}

