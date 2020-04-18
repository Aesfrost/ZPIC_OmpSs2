#include "particles.h"
#include "math.h"

#include <string.h>
#include <stdlib.h>
#include <openacc.h>

/*********************************************************************************************
 Particle Advance
 *********************************************************************************************/
#pragma acc routine seq
void interpolate_fld_openacc(const t_vfld *restrict const E, const t_vfld *restrict const B, const int nrow,
		const int ix, const int iy, const t_fld x, const t_fld y, t_vfld *restrict const Ep, t_vfld *restrict const Bp)
{
	const register int ih = ix + ((x < 0.5f) ? -1 : 0);
	const register int jh = iy + ((y < 0.5f) ? -1 : 0);

	const register t_fld w1h = x + ((x < 0.5f) ? 0.5f : -0.5f);
	const register t_fld w2h = y + ((y < 0.5f) ? 0.5f : -0.5f);


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

#pragma acc routine seq
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

void spec_kernel_openacc(t_species *spec, t_emf *emf, t_current *current)
{
	const t_part_data tem = 0.5 * spec->dt / spec->m_q;
	const t_part_data dt_dx = spec->dt / spec->dx[0];
	const t_part_data dt_dy = spec->dt / spec->dx[1];

	// Auxiliary values for current deposition
	t_part_data qnx = spec->q * spec->dx[0] / spec->dt;
	t_part_data qny = spec->q * spec->dx[1] / spec->dt;

	// Advance internal iteration number
	spec->iter += 1;
	spec->energy = 0;

	const int shift = spec->iter * spec->dt > spec->dx[0] * (spec->n_move + 1);
	const int size = emf->nrow * (emf->gc[1][0] + emf->nx[1] + emf->gc[1][1]);

	// Advance particles
	#pragma acc cache(tem, dt_dx, dt_dy, qnx, qny, shift, spec, emf->E_buf[0: size], emf->B_buf[0: size], current->J_buf[0: size])
	#pragma acc parallel loop independent //reduction(+ : spec->energy)
	for (int i = 0; i < spec->np; i++)
	{
		if(!spec->part.safe_to_delete[i])
		{
			t_vfld Ep, Bp;

			t_part_data utx, uty, utz, utsq;
			t_part_data gtem, otsq;

			t_part_data rg;
			t_part_data x1, y1;

			int di, dj;
			float dx, dy;

			// Interpolate fields
			interpolate_fld_openacc(emf->E, emf->B, emf->nrow, spec->part.ix[i], spec->part.iy[i],
					spec->part.x[i], spec->part.y[i], &Ep, &Bp);

			// Advance u using Boris scheme
			Ep.x *= tem;
			Ep.y *= tem;
			Ep.z *= tem;

			utx = spec->part.ux[i] + Ep.x;
			uty = spec->part.uy[i] + Ep.y;
			utz = spec->part.uz[i] + Ep.z;

			// Get time centered energy
			utsq = utx * utx + uty * uty + utz * utz;

	//		#pragma acc atomic
	//		spec->energy += (double) utsq / (sqrtf(1.0f + utsq) + 1);

			// Perform first half of the rotation
			gtem = tem / sqrtf(1.0f + utsq);

			Bp.x *= gtem;
			Bp.y *= gtem;
			Bp.z *= gtem;

			spec->part.ux[i] = utx + uty * Bp.z - utz * Bp.y;
			spec->part.uy[i] = uty + utz * Bp.x - utx * Bp.z;
			spec->part.uz[i] = utz + utx * Bp.y - uty * Bp.x;

			// Perform second half of the rotation
			otsq = 2.0f / (1.0f + Bp.x * Bp.x + Bp.y * Bp.y + Bp.z * Bp.z);

			Bp.x *= otsq;
			Bp.y *= otsq;
			Bp.z *= otsq;

			utx += spec->part.uy[i] * Bp.z - spec->part.uz[i] * Bp.y;
			uty += spec->part.uz[i] * Bp.x - spec->part.ux[i] * Bp.z;
			utz += spec->part.ux[i] * Bp.y - spec->part.uy[i] * Bp.x;

			// Perform second half of electric field acceleration
			spec->part.ux[i] = utx + Ep.x;
			spec->part.uy[i] = uty + Ep.y;
			spec->part.uz[i] = utz + Ep.z;

			// push particle
			rg = 1.0f / sqrtf(1.0f + spec->part.ux[i] * spec->part.ux[i]
									+ spec->part.uy[i] * spec->part.uy[i]
									+ spec->part.uz[i] * spec->part.uz[i]);

			dx = dt_dx * rg * spec->part.ux[i];
			dy = dt_dy * rg * spec->part.uy[i];

			x1 = spec->part.x[i] + dx;
			y1 = spec->part.y[i] + dy;

			di = (x1 >= 1.0f) - (x1 < 0.0f);
			dj = (y1 >= 1.0f) - (y1 < 0.0f);

			dep_current_openacc(spec->part.ix[i], spec->part.iy[i], di, dj, spec->part.x[i], spec->part.y[i],
					dx, dy, qnx, qny, spec->q * spec->part.uz[i] * rg, current->J, current->nrow);

			// Store results
			spec->part.x[i] = x1 - di;
			spec->part.y[i] = y1 - dj;
			spec->part.ix[i] += di;
			spec->part.iy[i] += dj;

			// First shift particle left (if applicable), then check for particles leaving the box
			if(spec->moving_window)
			{
				if (shift) spec->part.ix[i]--;

				if(spec->part.ix[i] < 0) spec->part.safe_to_delete[i] = true;
				else if(spec->part.ix[i] >= spec->nx[0]) spec->part.safe_to_delete[i] = true;

			}else
			{
				if(spec->part.ix[i] < 0) spec->part.ix[i] += spec->nx[0];
				else if(spec->part.ix[i] >= spec->nx[0]) spec->part.ix[i] -= spec->nx[0];
			}

			if(spec->part.iy[i] < 0) spec->part.iy[i] += spec->nx[1];
			else if(spec->part.iy[i] >= spec->nx[1]) spec->part.iy[i] -= spec->nx[1];
		}
	}
}

/*********************************************************************************************
 Particle Advance Post Processing
 *********************************************************************************************/

void spec_move_window_openacc(t_species *spec)
{
	if((spec->iter * spec->dt) > (spec->dx[0] * (spec->n_move + 1)))
	{
		// Increase moving window counter
		spec->n_move++;

		// Inject particles in the right edge of the simulation box
		const int range[][2] = {{spec->nx[0] - 1, spec->nx[0] - 1}, {0, spec->nx[1] - 1}};
		spec_inject_particles(spec, range);
	}
}

/*********************************************************************************************
 Sort
 *********************************************************************************************/

void spec_sort(t_species *spec, const int begin, const int end)
{
	#pragma acc set device_num(0)

	const int n_bins_x = spec->n_bins_x;
	const int n_bins_y = spec->n_bins_y;
	const int bin_size = spec->bin_size;

	int *temp = malloc(n_bins_y * n_bins_x * sizeof(int));
	int *count = malloc(n_bins_y * n_bins_x * sizeof(int));
	int *bin_idx = malloc(n_bins_y * n_bins_x * sizeof(int));

	t_part bins;
	bins.ix = malloc(spec->np_max * sizeof(*spec->part.ix));
	bins.iy = malloc(spec->np_max * sizeof(*spec->part.iy));
	bins.x = malloc(spec->np_max * sizeof(*spec->part.x));
	bins.y = malloc(spec->np_max * sizeof(*spec->part.y));
	bins.ux = malloc(spec->np_max * sizeof(*spec->part.ux));
	bins.uy = malloc(spec->np_max * sizeof(*spec->part.uy));
	bins.uz = malloc(spec->np_max * sizeof(*spec->part.uz));
	bins.safe_to_delete = NULL;

	int idx, ix, iy;

	memset(count, 0, n_bins_x * n_bins_y * sizeof(int));

	// Count the number of elements in each bin
	#pragma acc parallel loop independent private(ix, iy)
	for (int i = begin; i < end; i++)
		if (!spec->part.safe_to_delete[i])
		{
			ix = spec->part.ix[i] / bin_size;
			iy = spec->part.iy[i] / bin_size;

			#pragma acc atomic
			count[ix + iy * n_bins_x]++;
		}

	memcpy(bin_idx, count, spec->n_bins_y * spec->n_bins_x * sizeof(int));

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

	spec->np = bin_idx[n_bins_x * n_bins_y - 1];

	// Subtract the size to find the initial idx of each bin
	#pragma acc parallel loop independent
	for (int i = 0; i < n_bins_x * n_bins_y; i++)
		bin_idx[i] -= count[i];

	// Distribute the elements to the bins
	#pragma acc parallel loop independent private(idx, ix, iy)
	for (int i = begin; i < end; i++)
		if (!spec->part.safe_to_delete[i])
		{
			ix = spec->part.ix[i] / bin_size;
			iy = spec->part.iy[i] / bin_size;

			#pragma acc atomic capture
			{
				idx = bin_idx[ix + iy * n_bins_x];
				bin_idx[ix + iy * n_bins_x]++;
			}

			bins.ix[idx] = spec->part.ix[i];
			bins.iy[idx] = spec->part.iy[i];
			bins.x[idx] = spec->part.x[i];
			bins.y[idx] = spec->part.y[i];
			bins.ux[idx] = spec->part.ux[i];
			bins.uy[idx] = spec->part.uy[i];
			bins.uz[idx] = spec->part.uz[i];
		}

	#pragma acc parallel loop independent
	for (int k = 0; k < spec->np; k++)
	{
		spec->part.ix[begin + k] = bins.ix[k];
		spec->part.iy[begin + k] = bins.iy[k];
		spec->part.x[begin + k] = bins.x[k];
		spec->part.y[begin + k] = bins.y[k];
		spec->part.ux[begin + k] = bins.ux[k];
		spec->part.uy[begin + k] = bins.uy[k];
		spec->part.uz[begin + k] = bins.uz[k];
		spec->part.safe_to_delete[begin + k] = false;
	}

	// Cleaning
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

/*********************************************************************************************
 Wrapper
 *********************************************************************************************/

void spec_advance_openacc(t_species *spec, t_emf *emf, t_current *current)
{
	spec_kernel_openacc(spec, emf, current);

	// First shift particle left (if applicable), then check for particles leaving the box
	if(spec->moving_window) spec_move_window_openacc(spec);

	if(spec->iter % 15 == 0) spec_sort(spec, 0, spec->np);

}
