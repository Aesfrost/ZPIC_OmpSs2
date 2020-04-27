#include "particles.h"
#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*********************************************************************************************
 Particle Advance
 *********************************************************************************************/
void interpolate_fld_openacc(const t_vfld *restrict const E, const t_vfld *restrict const B, const int nrow,
		const int ix, const int iy, const t_fld x, const t_fld y, t_vfld *restrict const Ep, t_vfld *restrict const Bp)
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

void dep_current_openacc(int ix, int iy, int di, int dj, float x0, float y0, float dx, float dy,
		float qnx, float qny, float qvz, t_vfld *restrict const J, const int nrow)
{
	// Split the particle trajectory
	typedef struct {
		float x0, x1, y0, y1, dx, dy, qvz;
		int ix, iy;
	} t_vp;

	t_vp vp[3];
	int vnp = 1;

	// split
	vp[0].x0 = x0;
	vp[0].y0 = y0;

	vp[0].dx = dx;
	vp[0].dy = dy;

	vp[0].x1 = x0 + dx;
	vp[0].y1 = y0 + dy;

	vp[0].qvz = qvz / 2.0;

	vp[0].ix = ix;
	vp[0].iy = iy;

	// x split
	if (di != 0)
	{
		//int ib = ( di+1 )>>1;
		int ib = (di == 1);

		float delta = (x0 + dx - ib) / dx;

		// Add new particle
		vp[1].x0 = 1 - ib;
		vp[1].x1 = (x0 + dx) - di;
		vp[1].dx = dx * delta;
		vp[1].ix = ix + di;

		float ycross = y0 + dy * (1.0f - delta);

		vp[1].y0 = ycross;
		vp[1].y1 = vp[0].y1;
		vp[1].dy = dy * delta;
		vp[1].iy = iy;

		vp[1].qvz = vp[0].qvz * delta;

		// Correct previous particle
		vp[0].x1 = ib;
		vp[0].dx *= (1.0f - delta);

		vp[0].dy *= (1.0f - delta);
		vp[0].y1 = ycross;

		vp[0].qvz *= (1.0f - delta);

		vnp++;
	}

	// ysplit
	if (dj != 0)
	{
		int isy = 1 - (vp[0].y1 < 0.0f || vp[0].y1 >= 1.0f);

		// int jb = ( dj+1 )>>1;
		int jb = (dj == 1);

		// The static analyser gets confused by this but it is correct
		float delta = (vp[isy].y1 - jb) / vp[isy].dy;

		// Add new particle
		vp[vnp].y0 = 1 - jb;
		vp[vnp].y1 = vp[isy].y1 - dj;
		vp[vnp].dy = vp[isy].dy * delta;
		vp[vnp].iy = vp[isy].iy + dj;

		float xcross = vp[isy].x0 + vp[isy].dx * (1.0f - delta);

		vp[vnp].x0 = xcross;
		vp[vnp].x1 = vp[isy].x1;
		vp[vnp].dx = vp[isy].dx * delta;
		vp[vnp].ix = vp[isy].ix;

		vp[vnp].qvz = vp[isy].qvz * delta;

		// Correct previous particle
		vp[isy].y1 = jb;
		vp[isy].dy *= (1.0f - delta);

		vp[isy].dx *= (1.0f - delta);
		vp[isy].x1 = xcross;

		vp[isy].qvz *= (1.0f - delta);

		// Correct extra vp if needed
		if (isy < vnp - 1)
		{
			vp[1].y0 -= dj;
			vp[1].y1 -= dj;
			vp[1].iy += dj;
		}
		vnp++;
	}

	// Deposit virtual particle currents
	for (int k = 0; k < vnp; k++)
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

void spec_advance_openacc(t_species *restrict const spec, const t_emf *restrict const emf,
		t_current *restrict const current, const int limits_y[2])
{
	const t_part_data tem = 0.5 * spec->dt / spec->m_q;
	const t_part_data dt_dx = spec->dt / spec->dx[0];
	const t_part_data dt_dy = spec->dt / spec->dx[1];

	// Auxiliary values for current deposition
	const t_part_data qnx = spec->q * spec->dx[0] / spec->dt;
	const t_part_data qny = spec->q * spec->dx[1] / spec->dt;

	// Advance internal iteration number
	spec->iter += 1;
	spec->energy = 0;

	const int size = emf->nrow * (emf->gc[1][0] + emf->nx[1] + emf->gc[1][1]);

	// Advance particles
	#pragma acc cache(tem, dt_dx, dt_dy, qnx, qny, spec->main_vector, emf->E_buf[0: size], emf->B_buf[0: size], current->J_buf[0: size])
	#pragma acc parallel loop independent //reduction(+ : spec->energy)
	for (int i = 0; i < spec->main_vector.size; i++)
	{
		t_vfld Ep, Bp;

		t_part_data utx, uty, utz, utsq;
		t_part_data gtem, otsq;

		t_part_data rg;
		t_part_data x1, y1;

		int di, dj;
		float dx, dy;

		// Interpolate fields
		interpolate_fld_openacc(emf->E, emf->B, emf->nrow, spec->main_vector.ix[i],
				spec->main_vector.iy[i] - limits_y[0], spec->main_vector.x[i],
				spec->main_vector.y[i], &Ep, &Bp);

		// Advance u using Boris scheme
		Ep.x *= tem;
		Ep.y *= tem;
		Ep.z *= tem;

		utx = spec->main_vector.ux[i] + Ep.x;
		uty = spec->main_vector.uy[i] + Ep.y;
		utz = spec->main_vector.uz[i] + Ep.z;

		// Get time centered energy
		utsq = utx * utx + uty * uty + utz * utz;

		//		#pragma acc atomic
		//		spec->energy += (double) utsq / (sqrtf(1.0f + utsq) + 1);

		// Perform first half of the rotation
		gtem = tem / sqrtf(1.0f + utsq);

		Bp.x *= gtem;
		Bp.y *= gtem;
		Bp.z *= gtem;

		spec->main_vector.ux[i] = utx + uty * Bp.z - utz * Bp.y;
		spec->main_vector.uy[i] = uty + utz * Bp.x - utx * Bp.z;
		spec->main_vector.uz[i] = utz + utx * Bp.y - uty * Bp.x;

		// Perform second half of the rotation
		otsq = 2.0f / (1.0f + Bp.x * Bp.x + Bp.y * Bp.y + Bp.z * Bp.z);

		Bp.x *= otsq;
		Bp.y *= otsq;
		Bp.z *= otsq;

		utx += spec->main_vector.uy[i] * Bp.z - spec->main_vector.uz[i] * Bp.y;
		uty += spec->main_vector.uz[i] * Bp.x - spec->main_vector.ux[i] * Bp.z;
		utz += spec->main_vector.ux[i] * Bp.y - spec->main_vector.uy[i] * Bp.x;

		// Perform second half of electric field acceleration
		spec->main_vector.ux[i] = utx + Ep.x;
		spec->main_vector.uy[i] = uty + Ep.y;
		spec->main_vector.uz[i] = utz + Ep.z;

		// push particle
		rg = 1.0f / sqrtf(1.0f + spec->main_vector.ux[i] * spec->main_vector.ux[i]
								+ spec->main_vector.uy[i] * spec->main_vector.uy[i]
								+ spec->main_vector.uz[i] * spec->main_vector.uz[i]);

		dx = dt_dx * rg * spec->main_vector.ux[i];
		dy = dt_dy * rg * spec->main_vector.uy[i];

		x1 = spec->main_vector.x[i] + dx;
		y1 = spec->main_vector.y[i] + dy;

		di = (x1 >= 1.0f) - (x1 < 0.0f);
		dj = (y1 >= 1.0f) - (y1 < 0.0f);

		dep_current_openacc(spec->main_vector.ix[i], spec->main_vector.iy[i] - limits_y[0], di, dj,
				spec->main_vector.x[i], spec->main_vector.y[i], dx, dy, qnx, qny,
				spec->q * spec->main_vector.uz[i] * rg, current->J, current->nrow);

		// Store results
		spec->main_vector.x[i] = x1 - di;
		spec->main_vector.y[i] = y1 - dj;
		spec->main_vector.ix[i] += di;
		spec->main_vector.iy[i] += dj;
	}
}

/*********************************************************************************************
 Post Processing
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

// Post-processing for spec advance in the GPU, including sorting (bucket sort) and
// transferring particles to adjacent regions if needed.
void spec_post_processing_openacc(t_species *restrict spec, t_species *restrict const upper_spec,
		t_species *restrict const lower_spec, const int limits_y[2])
{
	#pragma acc set device_num(0)

	int iy, ix, idx;

	const bool shift = (spec->iter * spec->dt) > (spec->dx[0] * (spec->n_move + 1));

	const int nx0 = spec->nx[0];
	const int nx1 = spec->nx[1];

	const int n_bins_x = spec->n_bins_x;
	const int n_bins_y = spec->n_bins_y;

	int *restrict temp = malloc(n_bins_y * n_bins_x * sizeof(int));
	int *restrict count = malloc(n_bins_y * n_bins_x * sizeof(int));
	int *restrict bin_idx = malloc(n_bins_y * n_bins_x * sizeof(int));

	t_particle_vector *restrict const upper_buffer = &upper_spec->temp_buffer[0];
	t_particle_vector *restrict const lower_buffer = &lower_spec->temp_buffer[1];

//	openacc_sort_1(spec, count, bin_idx, n_bins_x, n_bins_y, limits_y);
//	openacc_sort_2(spec, lower_spec, upper_spec, count, bin_idx, n_bins_x, n_bins_y, limits_y);

	memset(count, 0, n_bins_x * n_bins_y * sizeof(int));

	#pragma acc cache(count[0: n_bins_x * n_bins_y])
	#pragma acc parallel loop independent private(ix, iy)
	for (int i = 0; i < spec->main_vector.size; i++)
	{
		// First shift particle left (if applicable), then check for particles leaving the box
		if (spec->moving_window)
		{
			if (shift) spec->main_vector.ix[i]--;

			if ((spec->main_vector.ix[i] >= 0) && (spec->main_vector.ix[i] < nx0))
			{
				if (spec->main_vector.iy[i] >= limits_y[0] && spec->main_vector.iy[i] < limits_y[1])
				{
					ix = spec->main_vector.ix[i] / BIN_SIZE;
					iy = spec->main_vector.iy[i] / BIN_SIZE;

					#pragma acc atomic
					count[ix + iy * n_bins_x]++;
				}
			}else spec->main_vector.safe_to_delete[i] = true;
		}else
		{
			//Periodic boundaries for both axis
			if (spec->main_vector.ix[i] < 0) spec->main_vector.ix[i] += nx0;
			else if (spec->main_vector.ix[i] >= nx0) spec->main_vector.ix[i] -= nx0;

			if (spec->main_vector.iy[i] >= limits_y[0] && spec->main_vector.iy[i] < limits_y[1])
			{
				ix = spec->main_vector.ix[i] / BIN_SIZE;
				iy = spec->main_vector.iy[i] / BIN_SIZE;

				#pragma acc atomic
				count[ix + iy * n_bins_x]++;
			}
		}
	}

	#pragma acc parallel loop
	for(int i = 0; i < spec->n_bins_y * spec->n_bins_x; i++)
		bin_idx[i] = count[i];

	// Prefix sum to find the initial index of each bin
	for (int n = 1; n < n_bins_x * n_bins_y; n *= 2)
	{
		#pragma acc parallel loop independent
		for (int i = 0; i < n_bins_x * n_bins_y - n; i++)
			temp[i] = bin_idx[i];

		#pragma acc parallel loop independent
		for (int i = n; i < n_bins_x * n_bins_y; i++)
			bin_idx[i] += temp[i - n];
	}

	#pragma acc set device_num(0)
	t_particle_vector bins;
	bins.ix = malloc(bin_idx[n_bins_x * n_bins_y - 1] * sizeof(int));
	bins.iy = malloc(bin_idx[n_bins_x * n_bins_y - 1] * sizeof(int));
	bins.x = malloc(bin_idx[n_bins_x * n_bins_y - 1] * sizeof(t_fld));
	bins.y = malloc(bin_idx[n_bins_x * n_bins_y - 1] * sizeof(t_fld));
	bins.ux = malloc(bin_idx[n_bins_x * n_bins_y - 1] * sizeof(t_fld));
	bins.uy = malloc(bin_idx[n_bins_x * n_bins_y - 1] * sizeof(t_fld));
	bins.uz = malloc(bin_idx[n_bins_x * n_bins_y - 1] * sizeof(t_fld));
	bins.safe_to_delete = NULL;

	// Subtract the size to find the initial idx of each bin
	#pragma acc parallel loop
	for (int i = 0; i < n_bins_x * n_bins_y; i++)
		bin_idx[i] -= count[i];

	// Distribute the particle in the main buffer to the bins
	#pragma acc cache(bin_idx[0: n_bins_x * n_bins_y], limits_y[0:2], lower_spec->temp_buffer, upper_spec->temp_buffer)
	#pragma acc parallel loop independent private(idx, ix, iy)
	for (int i = 0; i < spec->main_vector.size; i++)
	{
		if(!spec->main_vector.safe_to_delete[i])
		{
			iy = spec->main_vector.iy[i];

			//Verify if the particle is still in the correct region, and then, increase the particle count
			// in the respective bin. Otherwise, send the particle to the correct region
			if (iy < limits_y[0])
			{
				if (iy < 0) spec->main_vector.iy[i] += nx1;

				#pragma acc atomic capture
				{
					idx = lower_buffer->size;
					lower_buffer->size++;
				}

				lower_buffer->ix[idx] = spec->main_vector.ix[i];
				lower_buffer->iy[idx] = spec->main_vector.iy[i];
				lower_buffer->x[idx] = spec->main_vector.x[i];
				lower_buffer->y[idx] = spec->main_vector.y[i];
				lower_buffer->ux[idx] = spec->main_vector.ux[i];
				lower_buffer->uy[idx] = spec->main_vector.uy[i];
				lower_buffer->uz[idx] = spec->main_vector.uz[i];
				lower_buffer->safe_to_delete[idx] = false;

			} else if (iy >= limits_y[1])
			{
				if (iy >= nx1) spec->main_vector.iy[i] -= nx1;

				#pragma acc atomic capture
				{
					idx = upper_buffer->size;
					upper_buffer->size++;
				}

				upper_buffer->ix[idx] = spec->main_vector.ix[i];
				upper_buffer->iy[idx] = spec->main_vector.iy[i];
				upper_buffer->x[idx] = spec->main_vector.x[i];
				upper_buffer->y[idx] = spec->main_vector.y[i];
				upper_buffer->ux[idx] = spec->main_vector.ux[i];
				upper_buffer->uy[idx] = spec->main_vector.uy[i];
				upper_buffer->uz[idx] = spec->main_vector.uz[i];
				upper_buffer->safe_to_delete[idx] = false;

			} else
			{
				ix = spec->main_vector.ix[i] / BIN_SIZE;
				iy = spec->main_vector.iy[i] / BIN_SIZE;

				#pragma acc atomic capture
				{
					idx = bin_idx[ix + iy * n_bins_x];
					bin_idx[ix + iy * n_bins_x]++;
				}

				bins.ix[idx] = spec->main_vector.ix[i];
				bins.iy[idx] = spec->main_vector.iy[i];
				bins.x[idx] = spec->main_vector.x[i];
				bins.y[idx] = spec->main_vector.y[i];
				bins.ux[idx] = spec->main_vector.ux[i];
				bins.uy[idx] = spec->main_vector.uy[i];
				bins.uz[idx] = spec->main_vector.uz[i];
			}
		}
	}

//	update_spec_buffer_cpu(lower_buffer);
//	update_spec_buffer_cpu(upper_buffer);

	// Copy the elements back to the main buffer
	#pragma acc parallel loop
	for (int k = 0; k < bin_idx[n_bins_x * n_bins_y - 1]; k++)
	{
		spec->main_vector.ix[k] = bins.ix[k];
		spec->main_vector.iy[k] = bins.iy[k];
		spec->main_vector.x[k] = bins.x[k];
		spec->main_vector.y[k] = bins.y[k];
		spec->main_vector.ux[k] = bins.ux[k];
		spec->main_vector.uy[k] = bins.uy[k];
		spec->main_vector.uz[k] = bins.uz[k];
		spec->main_vector.safe_to_delete[k] = false;
	}

	spec->main_vector.size = bin_idx[n_bins_x * n_bins_y - 1];

	// Clean
	free(bins.ix);
	free(bins.iy);
	free(bins.x);
	free(bins.y);
	free(bins.ux);
	free(bins.uy);
	free(bins.uz);

	free(temp);
	free(count);
	free(bin_idx);
}

void spec_update_main_vector_openacc(t_species *restrict spec, const int limits_y[2])
{
	int idx;
	const int np_inj = spec->temp_buffer[0].size + spec->temp_buffer[1].size;

	if (spec->moving_window && (spec->iter * spec->dt) > (spec->dx[0] * (spec->n_move + 1)))
	{
		// Increase moving window counter
		spec->n_move++;

		// Inject particles in the right edge of the simulation box
		const int range[][2] = {{spec->nx[0] - 1, spec->nx[0]}, {limits_y[0], limits_y[1]}};

		spec_inject_particles(spec, range);
	}

	// Check if buffer is large enough and if not reallocate
	if (spec->main_vector.size + np_inj > spec->main_vector.size_max)
	{
		spec->main_vector.size_max = ((spec->main_vector.size_max + np_inj) / 1024 + 1) * 1024;
		realloc_vector(&spec->main_vector.ix, spec->main_vector.size, spec->main_vector.size_max, sizeof(int));
		realloc_vector(&spec->main_vector.iy, spec->main_vector.size, spec->main_vector.size_max, sizeof(int));
		realloc_vector(&spec->main_vector.x, spec->main_vector.size, spec->main_vector.size_max, sizeof(t_fld));
		realloc_vector(&spec->main_vector.y, spec->main_vector.size, spec->main_vector.size_max, sizeof(t_fld));
		realloc_vector(&spec->main_vector.ux, spec->main_vector.size, spec->main_vector.size_max, sizeof(t_fld));
		realloc_vector(&spec->main_vector.uy, spec->main_vector.size, spec->main_vector.size_max, sizeof(t_fld));
		realloc_vector(&spec->main_vector.uz, spec->main_vector.size, spec->main_vector.size_max, sizeof(t_fld));
		realloc_vector(&spec->main_vector.safe_to_delete, spec->main_vector.size, spec->main_vector.size_max, sizeof(bool));
	}

//	update_spec_buffer_gpu(&spec->main_vector);
//	update_spec_buffer_gpu(&spec->temp_buffer[0]);
//	update_spec_buffer_gpu(&spec->temp_buffer[1]);

	//Copy the particles from the temporary buffers to the main buffer
	for (int k = 0; k < 2; k++)
	{
		int size_temp = spec->temp_buffer[k].size;

		#pragma acc parallel loop
		for(int i = 0; i < size_temp; i++)
		{
			spec->main_vector.ix[i + spec->main_vector.size] = spec->temp_buffer[k].ix[i];
			spec->main_vector.iy[i + spec->main_vector.size] = spec->temp_buffer[k].iy[i];
			spec->main_vector.x[i + spec->main_vector.size] = spec->temp_buffer[k].x[i];
			spec->main_vector.y[i + spec->main_vector.size] = spec->temp_buffer[k].y[i];
			spec->main_vector.ux[i + spec->main_vector.size] = spec->temp_buffer[k].ux[i];
			spec->main_vector.uy[i + spec->main_vector.size] = spec->temp_buffer[k].uy[i];
			spec->main_vector.uz[i + spec->main_vector.size] = spec->temp_buffer[k].uz[i];
			spec->main_vector.safe_to_delete[i + spec->main_vector.size] = false;
		}

		spec->temp_buffer[k].size = 0;
		spec->main_vector.size += size_temp;
	}
}
