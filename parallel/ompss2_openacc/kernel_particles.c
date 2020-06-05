#include "particles.h"
#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*********************************************************************************************
 Particle Advance
 *********************************************************************************************/
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

	spec->energy = 0;

	const int size = emf->nrow * (emf->gc[1][0] + emf->nx[1] + emf->gc[1][1]);

	// Advance particles
	#pragma acc cache(tem, dt_dx, dt_dy, qnx, qny, spec->main_vector, emf->E_buf[0: size], emf->B_buf[0: size], current->J_buf[0: size])
	#pragma acc parallel loop independent
	for (int k = 0; k < spec->main_vector.size; k++)
	{
		if(!spec->main_vector.safe_to_delete[k])
		{
			t_vfld Ep, Bp;

			// Interpolate fields
			interpolate_fld_openacc(emf->E, emf->B, emf->nrow, spec->main_vector.ix[k],
					spec->main_vector.iy[k] - limits_y[0], spec->main_vector.x[k],
					spec->main_vector.y[k], &Ep, &Bp);

			// Advance u using Boris scheme
			Ep.x *= tem;
			Ep.y *= tem;
			Ep.z *= tem;

			t_part_data utx = spec->main_vector.ux[k] + Ep.x;
			t_part_data uty = spec->main_vector.uy[k] + Ep.y;
			t_part_data utz = spec->main_vector.uz[k] + Ep.z;

			// Get time centered energy
			t_part_data utsq = utx * utx + uty * uty + utz * utz;

			//		#pragma acc atomic
			//		spec->energy += (double) utsq / (sqrtf(1.0f + utsq) + 1);

			// Perform first half of the rotation
			t_part_data gtem = tem / sqrtf(1.0f + utsq);

			Bp.x *= gtem;
			Bp.y *= gtem;
			Bp.z *= gtem;

			spec->main_vector.ux[k] = utx + uty * Bp.z - utz * Bp.y;
			spec->main_vector.uy[k] = uty + utz * Bp.x - utx * Bp.z;
			spec->main_vector.uz[k] = utz + utx * Bp.y - uty * Bp.x;

			// Perform second half of the rotation
			t_part_data Bp_mag = Bp.x * Bp.x + Bp.y * Bp.y + Bp.z * Bp.z;
			t_part_data otsq = 2.0f / (1.0f + Bp_mag);

			Bp.x *= otsq;
			Bp.y *= otsq;
			Bp.z *= otsq;

			utx += spec->main_vector.uy[k] * Bp.z - spec->main_vector.uz[k] * Bp.y;
			uty += spec->main_vector.uz[k] * Bp.x - spec->main_vector.ux[k] * Bp.z;
			utz += spec->main_vector.ux[k] * Bp.y - spec->main_vector.uy[k] * Bp.x;

			// Perform second half of electric field acceleration
			spec->main_vector.ux[k] = utx + Ep.x;
			spec->main_vector.uy[k] = uty + Ep.y;
			spec->main_vector.uz[k] = utz + Ep.z;

			// Push particle
			t_part_data usq = spec->main_vector.ux[k] * spec->main_vector.ux[k]
					+ spec->main_vector.uy[k] * spec->main_vector.uy[k]
					+ spec->main_vector.uz[k] * spec->main_vector.uz[k];
			t_part_data rg = 1.0f / sqrtf(1.0f + usq);

			t_part_data dx = dt_dx * rg * spec->main_vector.ux[k];
			t_part_data dy = dt_dy * rg * spec->main_vector.uy[k];

			t_part_data x1 = spec->main_vector.x[k] + dx;
			t_part_data y1 = spec->main_vector.y[k] + dy;

			int di = (x1 >= 1.0f) - (x1 < 0.0f);
			int dj = (y1 >= 1.0f) - (y1 < 0.0f);

			t_part_data qvz = spec->q * spec->main_vector.uz[k] * rg;

			dep_current_openacc(spec->main_vector.ix[k], spec->main_vector.iy[k] - limits_y[0], di, dj,
					spec->main_vector.x[k], spec->main_vector.y[k], dx, dy, qnx, qny, qvz, current->J,
					current->nrow);

			// Store results
			spec->main_vector.x[k] = x1 - di;
			spec->main_vector.y[k] = y1 - dj;
			spec->main_vector.ix[k] += di;
			spec->main_vector.iy[k] += dj;
		}
	}

	// Advance internal iteration number
	spec->iter += 1;
}

/*********************************************************************************************
 Async Data Transfer
 *********************************************************************************************/
void update_spec_buffer_cpu(const t_particle_vector *restrict const vector)
{
	#pragma acc update self(vector->ix[0: vector->size]) async
	#pragma acc update self(vector->iy[0: vector->size]) async
	#pragma acc update self(vector->x[0: vector->size])	async
	#pragma acc update self(vector->y[0: vector->size])	async
	#pragma acc update self(vector->ux[0: vector->size]) async
	#pragma acc update self(vector->uy[0: vector->size]) async
	#pragma acc update self(vector->uz[0: vector->size]) async
	#pragma acc update self(vector->safe_to_delete[0: vector->size]) async
}

void update_spec_buffer_gpu(const t_particle_vector *restrict const vector)
{
	#pragma acc update device(vector->ix[0: vector->size]) async
	#pragma acc update device(vector->iy[0: vector->size]) async
	#pragma acc update device(vector->x[0: vector->size]) async
	#pragma acc update device(vector->y[0: vector->size]) async
	#pragma acc update device(vector->ux[0: vector->size]) async
	#pragma acc update device(vector->uy[0: vector->size]) async
	#pragma acc update device(vector->uz[0: vector->size]) async
	#pragma acc update device(vector->safe_to_delete[0: vector->size]) async
}

/*********************************************************************************************
 Post Processing 1 (Region Check)
 *********************************************************************************************/
// Check for particles leaving the domain
void spec_post_processing_1_openacc(t_species *restrict spec, t_species *restrict const upper_spec,
		t_species *restrict const lower_spec, const int limits_y[2])
{
	int iy, idx;

	const bool shift = (spec->iter * spec->dt) > (spec->dx[0] * (spec->n_move + 1));

	const int nx0 = spec->nx[0];
	const int nx1 = spec->nx[1];

	t_particle_vector *restrict const upper_buffer = &upper_spec->temp_buffer[0];
	t_particle_vector *restrict const lower_buffer = &lower_spec->temp_buffer[1];

	#pragma acc parallel loop private(iy, idx)
	for (int i = 0; i < spec->main_vector.size; i++)
	{
		if(!spec->main_vector.safe_to_delete[i])
		{

			if (spec->moving_window)
			{
				// Shift particles left
				if (shift) spec->main_vector.ix[i]--;

				// Verify if the particle is leaving the region
				if ((spec->main_vector.ix[i] >= 0) && (spec->main_vector.ix[i] < nx0))
				{
					iy = spec->main_vector.iy[i];

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

						spec->main_vector.safe_to_delete[i] = true;

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

						spec->main_vector.safe_to_delete[i] = true;

					}
				} else spec->main_vector.safe_to_delete[i] = true;
			} else
			{
				//Periodic boundaries for both axis
				if (spec->main_vector.ix[i] < 0) spec->main_vector.ix[i] += nx0;
				else if (spec->main_vector.ix[i] >= nx0) spec->main_vector.ix[i] -= nx0;

				iy = spec->main_vector.iy[i];

				// Check if the particle is leaving the box
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

					spec->main_vector.safe_to_delete[i] = true;

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

					spec->main_vector.safe_to_delete[i] = true;
				}
			}
		}
	}

	// Update the temp buffers of adjacent regions
	update_spec_buffer_cpu(lower_buffer);
	update_spec_buffer_cpu(upper_buffer);
}

/*********************************************************************************************
 Sort + Prefix Sum
 *********************************************************************************************/
void prefix_sum_openacc(int *restrict vector, const int size)
{
	#pragma acc set device_num(0)
	int *restrict temp = malloc(size * sizeof(int));

	for (int n = 1; n < size; n *= 2)
	{
		#pragma acc parallel loop firstprivate(n) async(20)
		for (int i = 0; i < size - n; i++)
			temp[i] = vector[i];

		#pragma acc parallel loop firstprivate(n) async(20)
		for (int i = n; i < size; i++)
			vector[i] += temp[i - n];
	}

	#pragma acc wait(20)
	free(temp);
}

// Bucket sort
void spec_sort_openacc(t_species *restrict spec, const int limits_y[2])
{
	#pragma acc set device_num(0)

	int iy, ix, idx;

	const int n_bins_x = spec->n_bins_x;
	const int n_bins_y = spec->n_bins_y;

	int *restrict count = malloc(n_bins_y * n_bins_x * sizeof(int));
	int *restrict bin_idx = malloc(n_bins_y * n_bins_x * sizeof(int));

	// Count the particles in each bin
	memset(count, 0, n_bins_x * n_bins_y * sizeof(int));

	#pragma acc cache(count[0: n_bins_x * n_bins_y])
	#pragma acc parallel loop private(ix, iy)
	for (int i = 0; i < spec->main_vector.size; i++)
	{
		if(!spec->main_vector.safe_to_delete[i])
		{
			ix = spec->main_vector.ix[i] / BIN_SIZE;
			iy = spec->main_vector.iy[i] / BIN_SIZE;

			#pragma acc atomic
			count[ix + iy * n_bins_x]++;
		}
	}

	// Copy the count to the bin_idx for prefix sum
	#pragma acc parallel loop
	for (int i = 0; i < spec->n_bins_y * spec->n_bins_x; i++)
		bin_idx[i] = count[i];

	// Prefix sum to find the initial idx of each bin
	prefix_sum_openacc(bin_idx, n_bins_x * n_bins_y);

	t_particle_vector bins;
	bins.ix = malloc(bin_idx[n_bins_x * n_bins_y - 1] * sizeof(int));
	bins.iy = malloc(bin_idx[n_bins_x * n_bins_y - 1] * sizeof(int));
	bins.x = malloc(bin_idx[n_bins_x * n_bins_y - 1] * sizeof(t_fld));
	bins.y = malloc(bin_idx[n_bins_x * n_bins_y - 1] * sizeof(t_fld));
	bins.ux = malloc(bin_idx[n_bins_x * n_bins_y - 1] * sizeof(t_fld));
	bins.uy = malloc(bin_idx[n_bins_x * n_bins_y - 1] * sizeof(t_fld));
	bins.uz = malloc(bin_idx[n_bins_x * n_bins_y - 1] * sizeof(t_fld));
	bins.safe_to_delete = NULL;

	int size = spec->main_vector.size;
	spec->main_vector.size = bin_idx[n_bins_x * n_bins_y - 1];

	// Subtract the size to find the initial idx of each bin
	#pragma acc parallel loop
	for (int i = 0; i < n_bins_x * n_bins_y; i++)
		bin_idx[i] -= count[i];

	// Distribute the particle in the main buffer to the bins
	#pragma acc cache(bin_idx[0: n_bins_x * n_bins_y], limits_y[0:2])
	#pragma acc parallel loop private(idx, ix, iy)
	for (int i = 0; i < size; i++)
	{
		if(!spec->main_vector.safe_to_delete[i])
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

	// Clean
	free(bins.ix);
	free(bins.iy);
	free(bins.x);
	free(bins.y);
	free(bins.ux);
	free(bins.uy);
	free(bins.uz);

	free(count);
	free(bin_idx);
}

/*********************************************************************************************
 Spec Clean (Remove invalid particles)
 *********************************************************************************************/

// Remove the invalid particles of the section of the array (that
// will be used to fill the rest of invalid particles)
void compact_vector_opeancc(t_particle_vector *restrict vector)
{
	#pragma acc set device_num(0)
	int *restrict new_pos = malloc(vector->size * sizeof(int));

	t_particle_vector old_part;
	old_part.ix = malloc(vector->size * sizeof(int));
	old_part.iy = malloc(vector->size * sizeof(int));
	old_part.x = malloc(vector->size * sizeof(t_fld));
	old_part.y = malloc(vector->size * sizeof(t_fld));
	old_part.ux = malloc(vector->size * sizeof(t_fld));
	old_part.uy = malloc(vector->size * sizeof(t_fld));
	old_part.uz = malloc(vector->size * sizeof(t_fld));
	old_part.safe_to_delete = malloc(vector->size * sizeof(bool));

	// Scan the vector to find the invalid particles
	if(vector->safe_to_delete[0]) new_pos[0] = -1;
	else new_pos[0] = 0;

	#pragma acc parallel loop
	for (int i = 1; i < vector->size; i++)
	{
		if(vector->safe_to_delete[i]) new_pos[i] = 0;
		else new_pos[i] = 1;
	}

	// Prefix sum to find the new idx of each particle
	prefix_sum_openacc(new_pos, vector->size);

	// Copy particles to a aux array
	#pragma acc parallel loop
	for (int i = 0; i < vector->size; i++)
	{
		old_part.ix[i] = vector->ix[i];
		old_part.iy[i] = vector->iy[i];
		old_part.x[i] = vector->x[i];
		old_part.y[i] = vector->y[i];
		old_part.ux[i] = vector->ux[i];
		old_part.uy[i] = vector->uy[i];
		old_part.uz[i] = vector->uz[i];
		old_part.safe_to_delete[i] = vector->safe_to_delete[i];
	}

	// Copy from the aux array back to the vector in
	// the new positions
	#pragma acc parallel loop
	for (int i = 0; i < vector->size; i++)
	{
		if (!old_part.safe_to_delete[i])
		{
			vector->ix[new_pos[i]] = old_part.ix[i];
			vector->iy[new_pos[i]] = old_part.iy[i];
			vector->x[new_pos[i]] = old_part.x[i];
			vector->y[new_pos[i]] = old_part.y[i];
			vector->ux[new_pos[i]] = old_part.ux[i];
			vector->uy[new_pos[i]] = old_part.uy[i];
			vector->uz[new_pos[i]] = old_part.uz[i];
			vector->safe_to_delete[new_pos[i]] = false;
		}
	}

	vector->size = new_pos[vector->size - 1] + 1;

	// Cleaning
	free(old_part.ix);
	free(old_part.iy);
	free(old_part.x);
	free(old_part.y);
	free(old_part.ux);
	free(old_part.uy);
	free(old_part.uz);
	free(old_part.safe_to_delete);

	free(new_pos);
}

// Identify the invalid particles, and fill the positions
// with elements in the end of the array
void spec_clean_vector_openacc(t_species *spec)
{
	int idx;

	// Count the invalid particles
	int count = 0;
	#pragma acc parallel loop reduction(+ : count)
	for(int i = 0; i < spec->main_vector.size; i++)
		if(spec->main_vector.safe_to_delete[i]) count++;

	if(count == 0) return;

	// Subsection of the array to be used to fill the invalid particles
	int start = spec->main_vector.size - count;

	t_particle_vector swap_vector;
	swap_vector.size = count;
	swap_vector.ix = spec->main_vector.ix + start;
	swap_vector.iy = spec->main_vector.iy + start;
	swap_vector.x = spec->main_vector.x + start;
	swap_vector.y = spec->main_vector.y + start;
	swap_vector.ux = spec->main_vector.ux + start;
	swap_vector.uy = spec->main_vector.uy + start;
	swap_vector.uz = spec->main_vector.uz + start;
	swap_vector.safe_to_delete = spec->main_vector.safe_to_delete + start;

	// If the subsection contains some invalid particles, remove them
	compact_vector_opeancc(&swap_vector);

	// Fill the remaining of invalid particles with the subsection
	count = 0;
	#pragma acc parallel loop private(idx)
	for (int i = 0; i < start; i++)
	{
		if(spec->main_vector.safe_to_delete[i])
		{
			#pragma acc atomic capture
			{
				idx = count;
				count++;
			}

			spec->main_vector.ix[i] = swap_vector.ix[idx];
			spec->main_vector.iy[i] = swap_vector.iy[idx];
			spec->main_vector.x[i] = swap_vector.x[idx];
			spec->main_vector.y[i] = swap_vector.y[idx];
			spec->main_vector.ux[i] = swap_vector.ux[idx];
			spec->main_vector.uy[i] = swap_vector.uy[idx];
			spec->main_vector.uz[i] = swap_vector.uz[idx];
			spec->main_vector.safe_to_delete[i] = false;
		}
	}

	spec->main_vector.size = start;
}

/*********************************************************************************************
 Post Processing 2 (Update main buffer + Move Window)
 *********************************************************************************************/
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
	int idx;
	int np_inj = spec->temp_buffer[0].size + spec->temp_buffer[1].size;

	// Update the temp buffers of the spec
	update_spec_buffer_gpu(&spec->temp_buffer[0]);
	update_spec_buffer_gpu(&spec->temp_buffer[1]);

	// Clean invalid particles in the array
	if(spec->iter % SORT_ITER != 0) spec_clean_vector_openacc(spec);

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
			realloc_vector(&spec->main_vector.ix, spec->main_vector.size, spec->main_vector.size_max, sizeof(int));
			realloc_vector(&spec->main_vector.iy, spec->main_vector.size, spec->main_vector.size_max, sizeof(int));
			realloc_vector(&spec->main_vector.x, spec->main_vector.size, spec->main_vector.size_max, sizeof(t_fld));
			realloc_vector(&spec->main_vector.y, spec->main_vector.size, spec->main_vector.size_max, sizeof(t_fld));
			realloc_vector(&spec->main_vector.ux, spec->main_vector.size, spec->main_vector.size_max, sizeof(t_fld));
			realloc_vector(&spec->main_vector.uy, spec->main_vector.size, spec->main_vector.size_max, sizeof(t_fld));
			realloc_vector(&spec->main_vector.uz, spec->main_vector.size, spec->main_vector.size_max, sizeof(t_fld));
			realloc_vector(&spec->main_vector.safe_to_delete, spec->main_vector.size, spec->main_vector.size_max, sizeof(bool));
		}

		spec_set_x_openacc(spec, range);
		spec_set_u_openacc(spec, start, spec->main_vector.size);

	}else
	{
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
