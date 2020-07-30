/*********************************************************************************************
 ZPIC
 kernel_particles.c

 Created by Nicolas Guidotti on 14/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "particles.h"
#include "math.h"
#include "utilities.h"

#define LOCAL_BUFFER_SIZE 1024
#define THREAD_BLOCK 320

#define MAX_VALUE(x, y) x > y ? x : y
#define MIN_VALUE(x, y) x < y ? x : y
#define LTRIM(x) (x >= 1.0f) - (x < 0.0f)

typedef struct {
	float x0, x1, y0, y1, dx, dy, qvz;
	int ix, iy;
} t_vp;

/*********************************************************************************************
 Utilities
 *********************************************************************************************/

void prefix_sum_openacc(int *restrict vector, const unsigned int size)
{
	const unsigned int num_blocks = ceil((float) size / LOCAL_BUFFER_SIZE);
	int *restrict block_sum = malloc(num_blocks * sizeof(int));

	#pragma acc parallel loop gang vector_length(LOCAL_BUFFER_SIZE / 2)
	for (int block_id = 0; block_id < num_blocks; block_id++)
	{
		const int begin_idx = block_id * LOCAL_BUFFER_SIZE;
		int local_buffer[LOCAL_BUFFER_SIZE];

		#pragma acc cache(local_buffer[0: LOCAL_BUFFER_SIZE])

		#pragma acc loop vector
		for (int i = 0; i < LOCAL_BUFFER_SIZE; i++)
		{
			if (i + begin_idx < size) local_buffer[i] = vector[i + begin_idx];
			else local_buffer[i] = 0;
		}

		for (int offset = 1; offset < LOCAL_BUFFER_SIZE; offset *= 2)
		{
			#pragma acc loop vector
			for (int i = offset - 1; i < LOCAL_BUFFER_SIZE; i += 2 * offset)
				local_buffer[i + offset] += local_buffer[i];

		}

		block_sum[block_id] = local_buffer[LOCAL_BUFFER_SIZE - 1];
		local_buffer[LOCAL_BUFFER_SIZE - 1] = 0;

		for (int offset = LOCAL_BUFFER_SIZE >> 1; offset > 0; offset >>= 1)
		{
			#pragma acc loop vector
			for (int i = offset - 1; i < LOCAL_BUFFER_SIZE; i += 2 * offset)
			{
				int temp = local_buffer[i];
				local_buffer[i] = local_buffer[i + offset];
				local_buffer[i + offset] += temp;
			}
		}

		#pragma acc loop vector
		for (int i = 0; i < LOCAL_BUFFER_SIZE; i++)
			if (i + begin_idx < size) vector[i + begin_idx] = local_buffer[i];
	}

	if(num_blocks > 1)
	{
		prefix_sum_openacc(block_sum, num_blocks);

		#pragma acc parallel loop gang
		for(int block_id = 1; block_id < num_blocks; block_id++)
		{
			const int begin_idx = block_id * LOCAL_BUFFER_SIZE;

			#pragma acc loop vector
			for(int i = 0; i < LOCAL_BUFFER_SIZE; i++)
				if(i + begin_idx < size) vector[i + begin_idx] += block_sum[block_id];
		}
	}

	free(block_sum);
}

void spec_move_vector_int(int *restrict vector, int *restrict source_idx, int *restrict target_idx, const int move_size)
{
	int *restrict temp = alloc_align_buffer(DEFAULT_ALIGNMENT, move_size * sizeof(t_integer2));

	if(source_idx)
	{
		#pragma acc parallel loop
		for(int i = 0; i < move_size; i++)
			temp[i] = vector[source_idx[i]];
	}else
	{
		#pragma acc parallel loop
		for(int i = 0; i < move_size; i++)
			temp[i] = vector[i];
	}

	#pragma acc parallel loop
	for(int i = 0; i < move_size; i++)
		if(target_idx[i] >= 0)
			vector[target_idx[i]] = temp[i];

	free_align_buffer(temp);
}

void spec_move_vector_float(float *restrict vector, int *restrict source_idx, int *restrict target_idx, const int move_size)
{
	float *restrict temp = alloc_align_buffer(DEFAULT_ALIGNMENT, move_size * sizeof(float));

	if(source_idx)
	{
		#pragma acc parallel loop
		for(int i = 0; i < move_size; i++)
			temp[i] = vector[source_idx[i]];
	}else
	{
		#pragma acc parallel loop
		for(int i = 0; i < move_size; i++)
			temp[i] = vector[i];
	}

	#pragma acc parallel loop
	for(int i = 0; i < move_size; i++)
		if(target_idx[i] >= 0)
			vector[target_idx[i]] = temp[i];

	free_align_buffer(temp);
}


/*********************************************************************************************
 Particle Advance
 *********************************************************************************************/

// EM fields interpolation. OpenAcc Task
void interpolate_fld_openacc(const t_vfld *restrict const E, const t_vfld *restrict const B,
		const int nrow, const int ix, const int iy, const t_fld x, const t_fld y,
		t_vfld *restrict const Ep, t_vfld *restrict const Bp)
{
	const int ih = ix + ((x < 0.5f) ? -1 : 0);
	const int jh = iy + ((y < 0.5f) ? -1 : 0);

	const t_fld w1h = x + ((x < 0.5f) ? 0.5f : -0.5f);
	const t_fld w2h = y + ((y < 0.5f) ? 0.5f : -0.5f);

	Ep->x = (E[ih + iy * nrow].x * (1.0f - w1h) + E[ih + 1 + iy * nrow].x * w1h) * (1.0f - y)
			+ (E[ih + (iy + 1) * nrow].x * (1.0f - w1h) + E[ih + 1 + (iy + 1) * nrow].x * w1h) * y;
	Ep->y = (E[ix + jh * nrow].y * (1.0f - x) + E[ix + 1 + jh * nrow].y * x) * (1.0f - w2h)
			+ (E[ix + (jh + 1) * nrow].y * (1.0f - x) + E[ix + 1 + (jh + 1) * nrow].y * x) * w2h;
	Ep->z = (E[ix + iy * nrow].z * (1.0f - x) + E[ix + 1 + iy * nrow].z * x) * (1.0f - y)
			+ (E[ix + (iy + 1) * nrow].z * (1.0f - x) + E[ix + 1 + (iy + 1) * nrow].z * x) * y;

	Bp->x = (B[ix + jh * nrow].x * (1.0f - x) + B[ix + 1 + jh * nrow].x * x) * (1.0f - w2h)
			+ (B[ix + (jh + 1) * nrow].x * (1.0f - x) + B[ix + 1 + (jh + 1) * nrow].x * x) * w2h;
	Bp->y = (B[ih + iy * nrow].y * (1.0f - w1h) + B[ih + 1 + iy * nrow].y * w1h) * (1.0f - y)
			+ (B[ih + (iy + 1) * nrow].y * (1.0f - w1h) + B[ih + 1 + (iy + 1) * nrow].y * w1h) * y;
	Bp->z = (B[ih + jh * nrow].z * (1.0f - w1h) + B[ih + 1 + jh * nrow].z * w1h) * (1.0f - w2h)
			+ (B[ih + (jh + 1) * nrow].z * (1.0f - w1h) + B[ih + 1 + (jh + 1) * nrow].z * w1h)
					* w2h;
}

// Current deposition (adapted Villasenor-Bunemann method). OpenAcc task
void dep_current_openacc(int ix, int iy, int di, int dj, float x0, float y0, float dx,
		float dy, float qnx, float qny, float qvz, t_vfld *restrict const J, const int nrow,
		t_vp vp[THREAD_BLOCK * 3], const int thread_id)
{
	const int begin = thread_id * 3;

	// Split the particle trajectory
	int vnp = 1;

	// split
	vp[begin].x0 = x0;
	vp[begin].y0 = y0;

	vp[begin].dx = dx;
	vp[begin].dy = dy;

	vp[begin].x1 = x0 + dx;
	vp[begin].y1 = y0 + dy;

	vp[begin].qvz = qvz / 2.0;

	vp[begin].ix = ix;
	vp[begin].iy = iy;

	// x split
	if (di != 0)
	{
		//int ib = ( di+1 )>>1;
		int ib = (di == 1);

		float delta = (x0 + dx - ib) / dx;

		// Add new particle
		vp[begin + 1].x0 = 1 - ib;
		vp[begin + 1].x1 = (x0 + dx) - di;
		vp[begin + 1].dx = dx * delta;
		vp[begin + 1].ix = ix + di;

		float ycross = y0 + dy * (1.0f - delta);

		vp[begin + 1].y0 = ycross;
		vp[begin + 1].y1 = vp[begin].y1;
		vp[begin + 1].dy = dy * delta;
		vp[begin + 1].iy = iy;

		vp[begin + 1].qvz = vp[begin].qvz * delta;

		// Correct previous particle
		vp[begin].x1 = ib;
		vp[begin].dx *= (1.0f - delta);

		vp[begin].dy *= (1.0f - delta);
		vp[begin].y1 = ycross;

		vp[begin].qvz *= (1.0f - delta);

		vnp++;
	}

	// ysplit
	if (dj != 0)
	{
		int isy = 1 - (vp[begin].y1 < 0.0f || vp[begin].y1 >= 1.0f);

		// int jb = ( dj+1 )>>1;
		int jb = (dj == 1);

		// The static analyser gets confused by this but it is correct
		float delta = (vp[begin + isy].y1 - jb) / vp[begin + isy].dy;

		// Add new particle
		vp[begin + vnp].y0 = 1 - jb;
		vp[begin + vnp].y1 = vp[begin + isy].y1 - dj;
		vp[begin + vnp].dy = vp[begin + isy].dy * delta;
		vp[begin + vnp].iy = vp[begin + isy].iy + dj;

		float xcross = vp[begin + isy].x0 + vp[begin + isy].dx * (1.0f - delta);

		vp[begin + vnp].x0 = xcross;
		vp[begin + vnp].x1 = vp[begin + isy].x1;
		vp[begin + vnp].dx = vp[begin + isy].dx * delta;
		vp[begin + vnp].ix = vp[begin + isy].ix;

		vp[begin + vnp].qvz = vp[begin + isy].qvz * delta;

		// Correct previous particle
		vp[begin + isy].y1 = jb;
		vp[begin + isy].dy *= (1.0f - delta);

		vp[begin + isy].dx *= (1.0f - delta);
		vp[begin + isy].x1 = xcross;

		vp[begin + isy].qvz *= (1.0f - delta);

		// Correct extra vp if needed
		if (isy < vnp - 1)
		{
			vp[begin + 1].y0 -= dj;
			vp[begin + 1].y1 -= dj;
			vp[begin + 1].iy += dj;
		}
		vnp++;
	}

	// Deposit virtual particle currents
	for (int k = begin; k < begin + vnp; k++)
	{
		float S0x[2], S1x[2], S0y[2], S1y[2];
		float wl1, wl2;
		float wp1[2], wp2[2];

		S0x[0] = 1.0f - vp[k].x0;
		S0x[1] = vp[k].x0;

		S1x[0] = 1.0f - vp[k].x1;
		S1x[1] = vp[k].x1;

		S0y[0] = 1.0f - vp[k].y0;
		S0y[1] = vp[k].y0;

		S1y[0] = 1.0f - vp[k].y1;
		S1y[1] = vp[k].y1;

		wl1 = qnx * vp[k].dx;
		wl2 = qny * vp[k].dy;

		wp1[0] = 0.5f * (S0y[0] + S1y[0]);
		wp1[1] = 0.5f * (S0y[1] + S1y[1]);

		wp2[0] = 0.5f * (S0x[0] + S1x[0]);
		wp2[1] = 0.5f * (S0x[1] + S1x[1]);

		#pragma acc atomic
		J[vp[k].ix + nrow * vp[k].iy].x += wl1 * wp1[0];

		#pragma acc atomic
		J[vp[k].ix + nrow * (vp[k].iy + 1)].x += wl1 * wp1[1];

		#pragma acc atomic
		J[vp[k].ix + nrow * vp[k].iy].y += wl2 * wp2[0];

		#pragma acc atomic
		J[vp[k].ix + 1 + nrow * vp[k].iy].y += wl2 * wp2[1];

		#pragma acc atomic
		J[vp[k].ix + nrow * vp[k].iy].z += vp[k].qvz
				* (S0x[0] * S0y[0] + S1x[0] * S1y[0] + (S0x[0] * S1y[0] - S1x[0] * S0y[0]) / 2.0f);

		#pragma acc atomic
		J[vp[k].ix + 1 + nrow * vp[k].iy].z += vp[k].qvz
				* (S0x[1] * S0y[0] + S1x[1] * S1y[0] + (S0x[1] * S1y[0] - S1x[1] * S0y[0]) / 2.0f);

		#pragma acc atomic
		J[vp[k].ix + nrow * (vp[k].iy + 1)].z += vp[k].qvz
				* (S0x[0] * S0y[1] + S1x[0] * S1y[1] + (S0x[0] * S1y[1] - S1x[0] * S0y[1]) / 2.0f);

		#pragma acc atomic
		J[vp[k].ix + 1 + nrow * (vp[k].iy + 1)].z += vp[k].qvz
				* (S0x[1] * S0y[1] + S1x[1] * S1y[1] + (S0x[1] * S1y[1] - S1x[1] * S0y[1]) / 2.0f);
	}
}

// Particle advance (OpenAcc)
void spec_advance_openacc_optimized(t_species *restrict const spec, const t_emf *restrict const emf,
		t_current *restrict const current, const int limits_y[2])
{
	const t_part_data tem = 0.5 * spec->dt / spec->m_q;
	const t_part_data dt_dx = spec->dt / spec->dx[0];
	const t_part_data dt_dy = spec->dt / spec->dx[1];

	// Auxiliary values for current deposition
	const t_part_data qnx = spec->q * spec->dx[0] / spec->dt;
	const t_part_data qny = spec->q * spec->dx[1] / spec->dt;

	const int nrow = emf->nrow;
	const int region_offset = limits_y[0];

	// Advance particles
	#pragma acc parallel loop gang collapse(2) vector_length(THREAD_BLOCK)
	for(int bin_y = 0; bin_y < spec->n_bins_y; bin_y++)
	{
		for(int bin_x = 0; bin_x < spec->n_bins_x; bin_x++)
		{
			const int bin_idx = bin_x + bin_y * spec->n_bins_x;
			const int begin = spec->bin_offset[bin_idx];
			const int end = spec->bin_offset[bin_idx + 1];

			t_vfld E[(BIN_SIZE + 2) * (BIN_SIZE + 2)];
			t_vfld B[(BIN_SIZE + 2) * (BIN_SIZE + 2)];
			t_vfld J[(BIN_SIZE + 3) * (BIN_SIZE + 3)];

			#pragma acc cache(E[0 : (BIN_SIZE + 2) * (BIN_SIZE + 2)])
			#pragma acc cache(B[0 : (BIN_SIZE + 2) * (BIN_SIZE + 2)])
			#pragma acc cache(J[0 : (BIN_SIZE + 3) * (BIN_SIZE + 3)])

			t_vp vp[THREAD_BLOCK * 3];
			#pragma acc cache(vp[0 : THREAD_BLOCK * 3])

			#pragma acc loop vector collapse(2)
			for(int j = 0; j < (BIN_SIZE + 2); j++)
			{
				for(int i = 0; i < (BIN_SIZE + 2); i++)
				{
					t_integer2 idx;
					idx.x = (bin_x * BIN_SIZE + i - 1);
					idx.y = (bin_y * BIN_SIZE + j - 1);

					E[i + j * (BIN_SIZE + 2)] = emf->E[idx.x + idx.y * nrow];
					B[i + j * (BIN_SIZE + 2)] = emf->B[idx.x + idx.y * nrow];
				}
			}

			#pragma acc loop vector
			for(int i = 0; i < (BIN_SIZE + 3) * (BIN_SIZE + 3); i++)
			{
				J[i].x = 0.0f;
				J[i].y = 0.0f;
				J[i].z = 0.0f;
			}

			#pragma acc loop vector
			for (int k = begin; k < end; k++)
			{
				register bool is_invalid = spec->main_vector.safe_to_delete[k];

				if(!is_invalid)
				{
					register t_float3 part_velocity;
					part_velocity.x = spec->main_vector.ux[k];
					part_velocity.y = spec->main_vector.uy[k];
					part_velocity.z = spec->main_vector.uz[k];

					register t_float2 part_pos;
					part_pos.x = spec->main_vector.x[k];
					part_pos.y = spec->main_vector.y[k];

					register t_integer2 part_idx;
					part_idx.x = spec->main_vector.ix[k] - (bin_x * BIN_SIZE - 1);
					part_idx.y = spec->main_vector.iy[k] - (bin_y * BIN_SIZE - 1) - region_offset;

					t_vfld Ep, Bp;

					// Interpolate fields
					interpolate_fld_openacc(E, B, (BIN_SIZE + 2), part_idx.x, part_idx.y, part_pos.x, part_pos.y, &Ep, &Bp);

					// Advance u using Boris scheme
					Ep.x *= tem;
					Ep.y *= tem;
					Ep.z *= tem;

					t_float3 ut;
					ut.x = part_velocity.x + Ep.x;
					ut.y = part_velocity.y + Ep.y;
					ut.z = part_velocity.z + Ep.z;

					// Perform first half of the rotation
					t_part_data ustq = ut.x * ut.x + ut.y * ut.y + ut.z * ut.z;
					t_part_data gtem = tem / sqrtf(1.0f + ustq);

					Bp.x *= gtem;
					Bp.y *= gtem;
					Bp.z *= gtem;

					part_velocity.x = ut.x + ut.y * Bp.z - ut.z * Bp.y;
					part_velocity.y = ut.y + ut.z * Bp.x - ut.x * Bp.z;
					part_velocity.z = ut.z + ut.x * Bp.y - ut.y * Bp.x;

					// Perform second half of the rotation
					t_part_data Bp_mag = Bp.x * Bp.x + Bp.y * Bp.y + Bp.z * Bp.z;
					t_part_data otsq = 2.0f / (1.0f + Bp_mag);

					Bp.x *= otsq;
					Bp.y *= otsq;
					Bp.z *= otsq;

					ut.x += part_velocity.y * Bp.z - part_velocity.z * Bp.y;
					ut.y += part_velocity.z * Bp.x - part_velocity.x * Bp.z;
					ut.z += part_velocity.x * Bp.y - part_velocity.y * Bp.x;

					// Perform second half of electric field acceleration
					part_velocity.x = ut.x + Ep.x;
					part_velocity.y = ut.y + Ep.y;
					part_velocity.z = ut.z + Ep.z;

					// Push particle
					t_part_data usq = part_velocity.x * part_velocity.x
							+ part_velocity.y * part_velocity.y + part_velocity.z * part_velocity.z;
					t_part_data rg = 1.0f / sqrtf(1.0f + usq);

					t_part_data dx = dt_dx * rg * part_velocity.x;
					t_part_data dy = dt_dy * rg * part_velocity.y;

					t_part_data x1 = part_pos.x + dx;
					t_part_data y1 = part_pos.y + dy;

					int di = LTRIM(x1);
					int dj = LTRIM(y1);

					t_part_data qvz = spec->q * part_velocity.z * rg;

					dep_current_openacc(part_idx.x, part_idx.y, di, dj, part_pos.x, part_pos.y, dx,
										dy, qnx, qny, qvz, J, (BIN_SIZE + 3), vp, k % THREAD_BLOCK);

					// Store results
					spec->main_vector.x[k] = x1 - di;
					spec->main_vector.y[k] = y1 - dj;
					spec->main_vector.ix[k] += di;
					spec->main_vector.iy[k] += dj;
					spec->main_vector.ux[k] = part_velocity.x;
					spec->main_vector.uy[k] = part_velocity.y;
					spec->main_vector.uz[k] = part_velocity.z;
				}
			}

			#pragma acc loop vector collapse(2)
			for(int j = 0; j < (BIN_SIZE + 3); j++)
			{
				for(int i = 0; i < (BIN_SIZE + 3); i++)
				{
					t_integer2 idx;
					idx.x = (bin_x * BIN_SIZE + i - 1);
					idx.y = (bin_y * BIN_SIZE + j - 1);

					#pragma acc atomic
					current->J[idx.x + idx.y * nrow].x += J[i + j * ((BIN_SIZE + 3))].x;

					#pragma acc atomic
					current->J[idx.x + idx.y * nrow].y += J[i + j * ((BIN_SIZE + 3))].y;

					#pragma acc atomic
					current->J[idx.x + idx.y * nrow].z += J[i + j * ((BIN_SIZE + 3))].z;
				}
			}
		}
	}

	// Advance internal iteration number
	spec->iter++;
}

// Particle advance (OpenAcc)
void spec_advance_openacc_default(t_species *restrict const spec, const t_emf *restrict const emf,
		t_current *restrict const current, const int limits_y[2])
{
	const t_part_data tem = 0.5 * spec->dt / spec->m_q;
	const t_part_data dt_dx = spec->dt / spec->dx[0];
	const t_part_data dt_dy = spec->dt / spec->dx[1];

	// Auxiliary values for current deposition
	const t_part_data qnx = spec->q * spec->dx[0] / spec->dt;
	const t_part_data qny = spec->q * spec->dx[1] / spec->dt;

	t_vfld *restrict E = emf->E;
	t_vfld *restrict B = emf->B;
	t_vfld *restrict J = current->J;

	const int nrow = emf->nrow;
	const int region_offset = limits_y[0];

	// Advance particles
	#pragma acc parallel loop
	for (int k = 0; k < spec->main_vector.size; k++)
	{
		register bool is_invalid = spec->main_vector.safe_to_delete[k];

		if (!is_invalid)
		{
			register t_float3 part_velocity;
			part_velocity.x = spec->main_vector.ux[k];
			part_velocity.y = spec->main_vector.uy[k];
			part_velocity.z = spec->main_vector.uz[k];

			register t_float2 part_pos;
			part_pos.x = spec->main_vector.x[k];
			part_pos.y = spec->main_vector.y[k];

			register t_integer2 part_idx;
			part_idx.x = spec->main_vector.ix[k];
			part_idx.y = spec->main_vector.iy[k] - region_offset;

			t_vfld Ep, Bp;

			// Interpolate fields
			interpolate_fld_openacc(E, B, nrow, part_idx.x, part_idx.y, part_pos.x,
									part_pos.y, &Ep, &Bp);

			// Advance u using Boris scheme
			Ep.x *= tem;
			Ep.y *= tem;
			Ep.z *= tem;

			t_float3 ut;
			ut.x = part_velocity.x + Ep.x;
			ut.y = part_velocity.y + Ep.y;
			ut.z = part_velocity.z + Ep.z;

			// Perform first half of the rotation
			t_part_data ustq = ut.x * ut.x + ut.y * ut.y + ut.z * ut.z;
			t_part_data gtem = tem / sqrtf(1.0f + ustq);

			Bp.x *= gtem;
			Bp.y *= gtem;
			Bp.z *= gtem;

			part_velocity.x = ut.x + ut.y * Bp.z - ut.z * Bp.y;
			part_velocity.y = ut.y + ut.z * Bp.x - ut.x * Bp.z;
			part_velocity.z = ut.z + ut.x * Bp.y - ut.y * Bp.x;

			// Perform second half of the rotation
			t_part_data Bp_mag = Bp.x * Bp.x + Bp.y * Bp.y + Bp.z * Bp.z;
			t_part_data otsq = 2.0f / (1.0f + Bp_mag);

			Bp.x *= otsq;
			Bp.y *= otsq;
			Bp.z *= otsq;

			ut.x += part_velocity.y * Bp.z - part_velocity.z * Bp.y;
			ut.y += part_velocity.z * Bp.x - part_velocity.x * Bp.z;
			ut.z += part_velocity.x * Bp.y - part_velocity.y * Bp.x;

			// Perform second half of electric field acceleration
			part_velocity.x = ut.x + Ep.x;
			part_velocity.y = ut.y + Ep.y;
			part_velocity.z = ut.z + Ep.z;

			// Push particle
			t_part_data usq = part_velocity.x * part_velocity.x + part_velocity.y * part_velocity.y
					+ part_velocity.z * part_velocity.z;
			t_part_data rg = 1.0f / sqrtf(1.0f + usq);

			t_part_data dx = dt_dx * rg * part_velocity.x;
			t_part_data dy = dt_dy * rg * part_velocity.y;

			t_part_data x1 = part_pos.x + dx;
			t_part_data y1 = part_pos.y + dy;

			int di = LTRIM(x1);
			int dj = LTRIM(y1);

			t_part_data qvz = spec->q * part_velocity.z * rg;

			t_vp vp[3];
			dep_current_openacc(part_idx.x, part_idx.y, di, dj, part_pos.x, part_pos.y, dx, dy, qnx,
								qny, qvz, J, nrow, vp, 0);

			// Store results
			spec->main_vector.x[k] = x1 - di;
			spec->main_vector.y[k] = y1 - dj;
			spec->main_vector.ix[k] += di;
			spec->main_vector.iy[k] += dj;
			spec->main_vector.ux[k] = part_velocity.x;
			spec->main_vector.uy[k] = part_velocity.y;
			spec->main_vector.uz[k] = part_velocity.z;
		}
	}

	// Advance internal iteration number
	spec->iter++;
}

void spec_advance_openacc(t_species *restrict const spec, const t_emf *restrict const emf,
		t_current *restrict const current, const int limits_y[2])
{
	if (spec->moving_window)
	{
		if (spec->density.type != UNIFORM && (spec->density.start / spec->dx[0] - spec->n_move) >= 0)
			spec_advance_openacc_default(spec, emf, current, limits_y);
		else spec_advance_openacc_optimized(spec, emf, current, limits_y);
	} else spec_advance_openacc_optimized(spec, emf, current, limits_y);
}

/*********************************************************************************************
 Async Data Transfer
 *********************************************************************************************/
//void update_spec_buffer_cpu(const t_particle_vector *restrict const vector)
//{
//	#pragma acc update self(vector->ix[0: vector->size]) async
//	#pragma acc update self(vector->iy[0: vector->size]) async
//	#pragma acc update self(vector->x[0: vector->size])	async
//	#pragma acc update self(vector->y[0: vector->size])	async
//	#pragma acc update self(vector->ux[0: vector->size]) async
//	#pragma acc update self(vector->uy[0: vector->size]) async
//	#pragma acc update self(vector->uz[0: vector->size]) async
//	#pragma acc update self(vector->safe_to_delete[0: vector->size]) async
//}
//
//void update_spec_buffer_gpu(const t_particle_vector *restrict const vector)
//{
//	#pragma acc update device(vector->ix[0: vector->size]) async
//	#pragma acc update device(vector->iy[0: vector->size]) async
//	#pragma acc update device(vector->x[0: vector->size]) async
//	#pragma acc update device(vector->y[0: vector->size]) async
//	#pragma acc update device(vector->ux[0: vector->size]) async
//	#pragma acc update device(vector->uy[0: vector->size]) async
//	#pragma acc update device(vector->uz[0: vector->size]) async
//	#pragma acc update device(vector->safe_to_delete[0: vector->size]) async
//}

/*********************************************************************************************
 Post Processing 1 (Region Check)
 *********************************************************************************************/

// Transfer particles between regions (if applicable). OpenAcc Task
void spec_post_processing_1_openacc(t_species *restrict spec, t_species *restrict const upper_spec,
		t_species *restrict const lower_spec, const int limits_y[2])
{
	const int num_blocks = ceil((float) spec->main_vector.size / LOCAL_BUFFER_SIZE);
	const bool shift = (spec->iter * spec->dt) > (spec->dx[0] * (spec->n_move + 1));
	const int nx0 = spec->nx[0];
	const int nx1 = spec->nx[1];

	t_particle_vector *restrict const upper_buffer = &upper_spec->temp_buffer[0];
	t_particle_vector *restrict const lower_buffer = &lower_spec->temp_buffer[1];

	#pragma acc parallel loop gang
	for(int k = 0; k < num_blocks; k++)
	{
		const int begin = k * LOCAL_BUFFER_SIZE;
		const int end = MIN_VALUE((begin + LOCAL_BUFFER_SIZE), spec->main_vector.size);
		const int size_batch = end - begin;

		bool is_invalid[LOCAL_BUFFER_SIZE];

		#pragma acc cache(spec->main_vector.ix[begin : LOCAL_BUFFER_SIZE])
		#pragma acc cache(spec->main_vector.iy[begin : LOCAL_BUFFER_SIZE])
		#pragma acc cache(spec->main_vector.safe_to_delete[begin : LOCAL_BUFFER_SIZE])
		#pragma acc cache(is_invalid[0 : LOCAL_BUFFER_SIZE])

		#pragma acc loop vector
		for(int i = 0; i < size_batch; i++)
			is_invalid[i] = spec->main_vector.safe_to_delete[begin + i];

		#pragma acc loop vector
		for(int i = 0; i < size_batch; i++)
		{
			if (!is_invalid[i])
			{
				int ix = spec->main_vector.ix[begin + i];
				int iy = spec->main_vector.iy[begin + i];

				if (spec->moving_window)
				{
					// Shift particles left
					if (shift) ix--;

					// Verify if the particle is leaving the region
					if ((ix < 0) || (ix >= nx0)) is_invalid[i] = true;  // Mark the particle as invalid
				} else
				{
					//Periodic boundaries for both axis
					if (ix < 0) ix += nx0;
					else if (ix >= nx0) ix -= nx0;
				}

				if(!is_invalid[i])
				{
					int idx;

					// Check if the particle is leaving the box
					if (iy < limits_y[0])
					{
						if (iy < 0) iy += nx1;

						// Reserve a position in the vector
						#pragma acc atomic capture
						idx = lower_buffer->size++;

						lower_buffer->ix[idx] = ix;
						lower_buffer->iy[idx] = iy;
						lower_buffer->x[idx] = spec->main_vector.x[begin + i];
						lower_buffer->y[idx] = spec->main_vector.y[begin + i];
						lower_buffer->ux[idx] = spec->main_vector.ux[begin + i];
						lower_buffer->uy[idx] = spec->main_vector.uy[begin + i];
						lower_buffer->uz[idx] = spec->main_vector.uz[begin + i];
						lower_buffer->safe_to_delete[idx] = false;

						is_invalid[i] = true;  // Mark the particle as invalid

					} else if (iy >= limits_y[1])
					{
						if (iy >= nx1) iy -= nx1;

						#pragma acc atomic capture
						idx = upper_buffer->size++;

						upper_buffer->ix[idx] = ix;
						upper_buffer->iy[idx] = iy;
						upper_buffer->x[idx] = spec->main_vector.x[begin + i];
						upper_buffer->y[idx] = spec->main_vector.y[begin + i];
						upper_buffer->ux[idx] = spec->main_vector.ux[begin + i];
						upper_buffer->uy[idx] = spec->main_vector.uy[begin + i];
						upper_buffer->uz[idx] = spec->main_vector.uz[begin + i];
						upper_buffer->safe_to_delete[idx] = false;

						is_invalid[i] = true;  // Mark the particle as invalid
					}
				}

				if(!is_invalid[i])
				{
					spec->main_vector.ix[begin + i] = ix;
					spec->main_vector.iy[begin + i] = iy;
				}

				spec->main_vector.safe_to_delete[begin + i] = is_invalid[i];
			}
		}
	}

	// Update the temp buffers of adjacent regions
//	update_spec_buffer_cpu(lower_buffer);
//	update_spec_buffer_cpu(upper_buffer);
}

/*********************************************************************************************
 Sort
 *********************************************************************************************/

// Bucket sort (no optimization)
void spec_sort_openacc_default(t_species *restrict spec, const int limits_y[2])
{
	int iy, ix;

	const int size = spec->main_vector.size;
	const int n_bins_x = spec->n_bins_x;
	const int n_bins_y = spec->n_bins_y;

	if(!spec->bin_offset) spec->bin_offset = malloc((n_bins_y * n_bins_x + 1) * sizeof(int));

	int *restrict bin_offset = spec->bin_offset;
	int *restrict pos = alloc_align_buffer(DEFAULT_ALIGNMENT, spec->main_vector.size * sizeof(int));

	// Count the particles in each bin
	#pragma acc parallel loop
	for(int i = 0; i <= n_bins_x * n_bins_y; i++)
		bin_offset[i] = 0;

	#pragma acc parallel loop private(ix, iy)
	for (int i = 0; i < size; i++)
	{
		if(!spec->main_vector.safe_to_delete[i])
		{
			ix = spec->main_vector.ix[i] / BIN_SIZE;
			iy = (spec->main_vector.iy[i] - limits_y[0]) / BIN_SIZE;

			#pragma acc atomic capture
			{
				pos[i] = bin_offset[ix + iy * n_bins_x];
				bin_offset[ix + iy * n_bins_x]++;
			}
		}else pos[i] = -1;
	}

	// Prefix sum to find the initial idx of each bin
	prefix_sum_openacc(bin_offset, n_bins_x * n_bins_y + 1);

	// Calculate the new position in the array
	#pragma acc parallel loop private(ix, iy)
	for (int i = 0; i < size; i++)
	{
		if (pos[i] >= 0)
		{
			ix = spec->main_vector.ix[i] / BIN_SIZE;
			iy = (spec->main_vector.iy[i] - limits_y[0]) / BIN_SIZE;

			pos[i] += bin_offset[ix + iy * n_bins_x];
		}
	}

	const int final_size = bin_offset[n_bins_x * n_bins_y];
	spec->main_vector.size = final_size;

	spec_move_vector_int(spec->main_vector.ix, NULL, pos, size);
	spec_move_vector_int(spec->main_vector.iy, NULL, pos, size);
	spec_move_vector_float(spec->main_vector.x, NULL, pos, size);
	spec_move_vector_float(spec->main_vector.y, NULL, pos, size);
	spec_move_vector_float(spec->main_vector.ux, NULL, pos, size);
	spec_move_vector_float(spec->main_vector.uy, NULL, pos, size);
	spec_move_vector_float(spec->main_vector.uz, NULL, pos, size);

	free_align_buffer(pos); // Clean position vector

	#pragma acc parallel loop
	for (int k = 0; k < final_size; k++)
		spec->main_vector.safe_to_delete[k] = false;
}

// Bucket sort (assuming that particles are almost sorted)
void spec_sort_openacc_almost(t_species *restrict spec, const int limits_y[2])
{
	const int offset_region = limits_y[0];
	const int size = spec->main_vector.size;
	const int n_bins_x = spec->n_bins_x;
	const int n_bins = spec->n_bins_x * spec->n_bins_y;

	if(!spec->bin_offset) spec->bin_offset = malloc((n_bins + 1) * sizeof(int));

	int *restrict bin_offset = spec->bin_offset;
	int *restrict new_bin_offset = malloc((n_bins + 1) * sizeof(int));
	int *restrict part_target_bin = alloc_align_buffer(DEFAULT_ALIGNMENT, size * sizeof(int));

	#pragma acc parallel loop
	for (int i = 0; i < n_bins + 1; i++)
		new_bin_offset[i] = 0;

	#pragma acc parallel loop gang
	for (int i = 0; i < n_bins + 1; i++)
	{
		const register int begin = bin_offset[i];
		const register int end = i == n_bins ? size : bin_offset[i + 1];
		int count = 0;

		#pragma acc loop vector reduction(+ : count)
		for(int k = begin; k < end; k++)
		{
			register int target_bin;
			register int ix = spec->main_vector.ix[k] / BIN_SIZE;
			register int iy = (spec->main_vector.iy[k] - offset_region) / BIN_SIZE;
			register bool is_invalid = spec->main_vector.safe_to_delete[k];

			if(!is_invalid)
			{
				target_bin = ix + iy * n_bins_x;

				if(target_bin == i) count++;
				else
				{
					#pragma acc atomic
					new_bin_offset[target_bin]++;
				}
			}else target_bin = -1;

			part_target_bin[k] = target_bin;
		}

		#pragma acc atomic
		new_bin_offset[i] += count;
	}

	// Prefix sum to find the initial idx of each bin
	prefix_sum_openacc(new_bin_offset, n_bins + 1);
	spec->main_vector.size = new_bin_offset[n_bins];
	spec->bin_offset[n_bins] = new_bin_offset[n_bins];

	int *restrict leaving_offset = calloc((n_bins + 1), sizeof(int));

	#pragma acc parallel loop gang
	for (int i = 0; i < n_bins; i++)
	{
		const int begin = new_bin_offset[i];
		const int end = new_bin_offset[i + 1];
		int leaving_count = 0;

		spec->bin_offset[i] = begin;

		#pragma acc loop vector reduction(+ : leaving_count)
		for(int k = begin; k < end; k++)
		{
			register int target_bin = part_target_bin[k];
			if(target_bin != i) leaving_count++;
		}

		leaving_offset[i] = leaving_count;
	}

	free(new_bin_offset);

	int count = 0;

	#pragma acc parallel loop reduction(+ : count)
	for(int i = spec->main_vector.size; i < size; i++)
		if(spec->main_vector.safe_to_delete[i]) count++;

	prefix_sum_openacc(leaving_offset, n_bins + 1);
	const int temp_buffer_size = leaving_offset[n_bins] + count;

	int *restrict source_idx = malloc(temp_buffer_size * sizeof(int));
	int *restrict target_idx = malloc(temp_buffer_size * sizeof(int));
	int *restrict holes_idx = malloc(temp_buffer_size * sizeof(int));
	int idx_buffer_counter = 0;

	#pragma acc parallel loop gang
	for (int i = 0; i < n_bins; i++)
	{
		const int begin = bin_offset[i];
		const int end = bin_offset[i + 1];
		int offset = leaving_offset[i];

		#pragma acc loop vector
		for(int k = begin; k < end; k++)
		{
			register int idx;
			register int target_bin = part_target_bin[k];

			if(target_bin != i)
			{
				#pragma acc atomic capture
				idx = offset++;

				holes_idx[idx] = k;

				if (target_bin >= 0)
				{
					#pragma acc atomic capture
					idx = idx_buffer_counter++;

					source_idx[idx] = k;
				}
			}
		}
	}

	#pragma acc parallel loop
	for (int i = spec->main_vector.size; i < size; i++)
	{
		int idx = idx_buffer_counter + i - spec->main_vector.size;
		source_idx[idx] = i;
	}

	#pragma acc parallel loop
	for(int i = 0; i < temp_buffer_size; i++)
	{
		int idx;
		const register int source = source_idx[i];
		register int target_bin = part_target_bin[source];

		if (target_bin >= 0)
		{
			#pragma acc atomic capture
			idx = leaving_offset[target_bin]++;

			target_idx[i] = holes_idx[idx];
		}else target_idx[i] = -1;
	}

	free(holes_idx);
	free(leaving_offset);
	free_align_buffer(part_target_bin);

	spec_move_vector_int(spec->main_vector.ix, source_idx, target_idx, temp_buffer_size);
	spec_move_vector_int(spec->main_vector.iy, source_idx, target_idx, temp_buffer_size);
	spec_move_vector_float(spec->main_vector.x, source_idx, target_idx, temp_buffer_size);
	spec_move_vector_float(spec->main_vector.y, source_idx, target_idx, temp_buffer_size);
	spec_move_vector_float(spec->main_vector.ux, source_idx, target_idx, temp_buffer_size);
	spec_move_vector_float(spec->main_vector.uy, source_idx, target_idx, temp_buffer_size);
	spec_move_vector_float(spec->main_vector.uz, source_idx, target_idx, temp_buffer_size);

	free(source_idx);
	free(target_idx);

	#pragma acc parallel loop
	for (int k = 0; k < size; k++)
		spec->main_vector.safe_to_delete[k] = false;
}


void spec_sort_openacc(t_species *restrict spec, const int limits_y[2])
{
	if(spec->moving_window && spec->density.type != UNIFORM)
	{
		if(spec->density.start / spec->dx[0] - spec->n_move < 0) spec_sort_openacc_almost(spec, limits_y);
	}else
	{
		if(spec->iter == 0)
			spec_sort_openacc_default(spec, limits_y);
		else spec_sort_openacc_almost(spec, limits_y);
	}
}

/*********************************************************************************************
 Post Processing 2 (Update main buffer + Move Window)
 *********************************************************************************************/
// Reset the particle velocity (OpenAcc)
void spec_set_u_openacc(t_species *spec, const int start, const int end)
{
	#pragma acc parallel loop
	for (int i = start; i <= end; i++)
	{
		spec->main_vector.ux[i] = 0;
		spec->main_vector.uy[i] = 0;
		spec->main_vector.uz[i] = 0;
	}
}

// Set the position of the injected particles (OpenAcc)
void spec_set_x_openacc(t_species *spec, const int range[][2])
{
	int ip;

	float *poscell;
	int start, end;

	// Calculate particle positions inside the cell
	const int npc = spec->ppc[0] * spec->ppc[1];
	t_part_data const dpcx = 1.0f / spec->ppc[0];
	t_part_data const dpcy = 1.0f / spec->ppc[1];

	poscell = malloc(2 * npc * sizeof(t_part_data));
	ip = 0;

	for (int j = 0; j < spec->ppc[1]; j++)
	{
		for (int i = 0; i < spec->ppc[0]; i++)
		{
			poscell[ip] = dpcx * (i + 0.5);
			poscell[ip + 1] = dpcy * (j + 0.5);
			ip += 2;
		}
	}

	ip = spec->main_vector.size;

	// Set position of particles in the specified grid range according to the density profile
	switch (spec->density.type)
	{
		case STEP:    // Step like density profile

			// Get edge position normalized to cell size;
			start = spec->density.start / spec->dx[0] - spec->n_move;
			if(range[0][0] > start) start = range[0][0];

			end = range[0][1];

			#pragma acc parallel loop independent collapse(3)
			for (int j = range[1][0]; j < range[1][1]; j++)
				for (int i = start; i < end; i++)
					for (int k = 0; k < npc; k++)
					{
						int idx = (i - start) * npc + (j - range[1][0]) * (end - start) * npc + k + ip;

						spec->main_vector.ix[idx] = i;
						spec->main_vector.iy[idx] = j;
						spec->main_vector.x[idx] = poscell[2 * k];
						spec->main_vector.y[idx] = poscell[2 * k + 1];
						spec->main_vector.safe_to_delete[idx] = false;
					}

			break;

		case SLAB:    // Slab like density profile

			// Get edge position normalized to cell size;
			start = spec->density.start / spec->dx[0] - spec->n_move;
			end = spec->density.end / spec->dx[0] - spec->n_move;

			if(start < range[0][0]) start = range[0][0];
			if(end > range[0][1]) end = range[0][1];

			#pragma acc parallel loop independent collapse(3)
			for (int j = range[1][0]; j < range[1][1]; j++)
				for (int i = start; i < end; i++)
					for (int k = 0; k < npc; k++)
					{
						int idx = (i - start) * npc + (j - range[1][0]) * (end - start) * npc + k + ip;

						spec->main_vector.ix[idx] = i;
						spec->main_vector.iy[idx] = j;
						spec->main_vector.x[idx] = poscell[2 * k];
						spec->main_vector.y[idx] = poscell[2 * k + 1];
						spec->main_vector.safe_to_delete[idx] = false;
					}
			break;

		default:    // Uniform density
			start = range[0][0];
			end = range[0][1];

			#pragma acc parallel loop independent collapse(3)
			for (int j = range[1][0]; j < range[1][1]; j++)
				for (int i = start; i < end; i++)
					for (int k = 0; k < npc; k++)
					{
						int idx = (i - start) * npc + (j - range[1][0]) * (end - start) * npc + k + ip;

						spec->main_vector.ix[idx] = i;
						spec->main_vector.iy[idx] = j;
						spec->main_vector.x[idx] = poscell[2 * k];
						spec->main_vector.y[idx] = poscell[2 * k + 1];
						spec->main_vector.safe_to_delete[idx] = false;
					}
				}

	spec->main_vector.size += (range[1][1] - range[1][0]) * (end - start) * npc;

	free(poscell);
}

void spec_post_processing_2_openacc(t_species *restrict spec, const int limits_y[2])
{
	int np_inj = spec->temp_buffer[0].size + spec->temp_buffer[1].size;

//	printf("Invalid: %d\n Temp Buffer Size: %d\n", spec->invalid_count, np_inj);

//	// Update the temp buffers of the spec
//	update_spec_buffer_gpu(&spec->temp_buffer[0]);
//	update_spec_buffer_gpu(&spec->temp_buffer[1]);

	// Move window
	if (spec->moving_window && (spec->iter * spec->dt) > (spec->dx[0] * (spec->n_move + 1)))
	{
		const int start = spec->main_vector.size;

		// Increase moving window counter
		spec->n_move++;

		// Inject particles in the right edge of the simulation box
		const int range[][2] = {{spec->nx[0] - 1, spec->nx[0]}, { limits_y[0], limits_y[1]}};
		np_inj += (range[0][1] - range[0][0]) * (range[1][1] - range[1][0]) * spec->ppc[0] * spec->ppc[1];

		// Check if buffer is large enough and if not reallocate
		if (spec->main_vector.size + np_inj > spec->main_vector.size_max)
		{
			spec->main_vector.size_max = ((spec->main_vector.size_max + np_inj) / 1024 + 1) * 1024;
			realloc_align_buffer((void **) &spec->main_vector.ix, spec->main_vector.size, spec->main_vector.size_max,
									sizeof(int), DEFAULT_ALIGNMENT);
			realloc_align_buffer((void **) &spec->main_vector.iy, spec->main_vector.size, spec->main_vector.size_max,
									sizeof(int), DEFAULT_ALIGNMENT);
			realloc_align_buffer((void **) &spec->main_vector.x, spec->main_vector.size, spec->main_vector.size_max,
									sizeof(t_fld), DEFAULT_ALIGNMENT);
			realloc_align_buffer((void **) &spec->main_vector.y, spec->main_vector.size, spec->main_vector.size_max,
									sizeof(t_fld), DEFAULT_ALIGNMENT);
			realloc_align_buffer((void **) &spec->main_vector.ux, spec->main_vector.size, spec->main_vector.size_max,
									sizeof(t_fld), DEFAULT_ALIGNMENT);
			realloc_align_buffer((void **) &spec->main_vector.uy, spec->main_vector.size, spec->main_vector.size_max,
									sizeof(t_fld), DEFAULT_ALIGNMENT);
			realloc_align_buffer((void **) &spec->main_vector.uz, spec->main_vector.size, spec->main_vector.size_max,
									sizeof(t_fld), DEFAULT_ALIGNMENT);
			realloc_align_buffer((void **) &spec->main_vector.safe_to_delete, spec->main_vector.size,
									spec->main_vector.size_max, sizeof(bool), DEFAULT_ALIGNMENT);
		}

		spec_set_x_openacc(spec, range);
		spec_set_u_openacc(spec, start, spec->main_vector.size);

	}else
	{
		// Check if buffer is large enough and if not reallocate
		if (spec->main_vector.size + np_inj > spec->main_vector.size_max)
		{
			spec->main_vector.size_max = ((spec->main_vector.size_max + np_inj) / 1024 + 1) * 1024;
			realloc_align_buffer((void **) &spec->main_vector.ix, spec->main_vector.size, spec->main_vector.size_max,
									sizeof(int), DEFAULT_ALIGNMENT);
			realloc_align_buffer((void **) &spec->main_vector.iy, spec->main_vector.size, spec->main_vector.size_max,
									sizeof(int), DEFAULT_ALIGNMENT);
			realloc_align_buffer((void **) &spec->main_vector.x, spec->main_vector.size, spec->main_vector.size_max,
									sizeof(t_fld), DEFAULT_ALIGNMENT);
			realloc_align_buffer((void **) &spec->main_vector.y, spec->main_vector.size, spec->main_vector.size_max,
									sizeof(t_fld), DEFAULT_ALIGNMENT);
			realloc_align_buffer((void **) &spec->main_vector.ux, spec->main_vector.size, spec->main_vector.size_max,
									sizeof(t_fld), DEFAULT_ALIGNMENT);
			realloc_align_buffer((void **) &spec->main_vector.uy, spec->main_vector.size, spec->main_vector.size_max,
									sizeof(t_fld), DEFAULT_ALIGNMENT);
			realloc_align_buffer((void **) &spec->main_vector.uz, spec->main_vector.size, spec->main_vector.size_max,
									sizeof(t_fld), DEFAULT_ALIGNMENT);
			realloc_align_buffer((void **) &spec->main_vector.safe_to_delete, spec->main_vector.size,
									spec->main_vector.size_max, sizeof(bool), DEFAULT_ALIGNMENT);
		}
	}

	//Copy the particles from the temporary buffers to the main buffer
	for (int k = 0; k < 2; k++)
	{
		int size_temp = spec->temp_buffer[k].size;
		int size = spec->main_vector.size;

		#pragma acc parallel loop firstprivate(size, size_temp)
		for (int i = 0; i < size_temp; i++)
		{
			spec->main_vector.ix[i + size] = spec->temp_buffer[k].ix[i];
			spec->main_vector.iy[i + size] = spec->temp_buffer[k].iy[i];
			spec->main_vector.x[i + size] = spec->temp_buffer[k].x[i];
			spec->main_vector.y[i + size] = spec->temp_buffer[k].y[i];
			spec->main_vector.ux[i + size] = spec->temp_buffer[k].ux[i];
			spec->main_vector.uy[i + size] = spec->temp_buffer[k].uy[i];
			spec->main_vector.uz[i + size] = spec->temp_buffer[k].uz[i];
			spec->main_vector.safe_to_delete[i + size] = false;
		}

		spec->temp_buffer[k].size = 0;
		spec->main_vector.size += size_temp;
	}
}
