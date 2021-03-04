/*
 *  particles.c
 *  zpic
 *
 *  Created by Ricardo Fonseca on 11/8/10.
 *  Copyright 2010 Centro de Física dos Plasmas. All rights reserved.
 *
 */

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#include "particles.h"

#include "random.h"
#include "emf.h"
#include "current.h"

#include "zdf.h"
#include "timer.h"

#include "omp.h"

static double _spec_time = 0.0;
static double _spec_npush = 0.0;

/**
 * OpenMP reduction function auxiliaries
 */

typedef struct {
	t_vfld *J;
	int J_buff_size;
	int J_size;
} t_reduce;

t_reduce *reduce_array = NULL;

void alloc(int buff_size, int size)
{
	int num_threads = omp_get_num_threads();

	reduce_array = malloc(num_threads * sizeof(t_reduce));
	assert(reduce_array);

	for (int i = 0; i < num_threads; i++)
	{
		reduce_array[i].J = malloc(buff_size * sizeof(t_vfld));
		assert(reduce_array[i].J);
		reduce_array[i].J += (buff_size - size);
		reduce_array[i].J_buff_size = buff_size;
		reduce_array[i].J_size = size;
	}
}

void dealloc()
{
	int num_threads = omp_get_num_threads();
	for (int i = 0; i < num_threads; i++)
	{
		free(reduce_array[i].J - (reduce_array[i].J_buff_size - reduce_array[i].J_size));
	}
	free(reduce_array);
}

t_reduce init_tvfld(t_reduce reduce_orig)
{
	t_reduce reduce_priv = reduce_array[omp_get_thread_num()];

	const int buff_size = reduce_orig.J_buff_size;
	const int size = reduce_orig.J_size;

	/* In doubt see current_zero function in current.c */
	size_t reset_size = buff_size * sizeof(t_vfld);
	memset(reduce_priv.J - (buff_size - size), 0, reset_size);

	return reduce_priv;
}

void add_tvfld(t_reduce reduce_out, t_reduce reduce_in)
{
	const int size = reduce_out.J_size;
	const int start = reduce_out.J_buff_size - reduce_out.J_size;

	for (int i = -start; i < size; i++)
	{
		reduce_out.J[i].x += reduce_in.J[i].x;
		reduce_out.J[i].y += reduce_in.J[i].y;
		reduce_out.J[i].z += reduce_in.J[i].z;
	}
}

t_reduce reduce;

#pragma omp declare reduction(tvfldAdd: t_reduce: add_tvfld(omp_out, omp_in)) \
				initializer( omp_priv = init_tvfld(omp_orig))

void spec_sort(t_species *spec);

/**
 * Returns the total time spent pushing particles (includes boundaries and moving window)
 * @return  Total time in seconds
 */
double spec_time(void)
{
	return _spec_time;
}

/**
 * Returns the performance achieved by the code (push time)
 * @return  Performance in seconds per particle
 */
double spec_perf(void)
{
	return (_spec_npush > 0) ? _spec_time / _spec_npush : 0.0;
}

/*********************************************************************************************
 Initialization
 *********************************************************************************************/

// Set the momentum of the injected particles
void spec_set_u(t_part *vector, const int start, const int end, const t_part_data ufl[3],
		const t_part_data uth[3])
{
	for (int i = start; i < end; i++)
	{
		vector[i].ux = ufl[0] + uth[0] * rand_norm();
		vector[i].uy = ufl[1] + uth[1] * rand_norm();
		vector[i].uz = ufl[2] + uth[2] * rand_norm();
	}
}

// Set the initial position of the particles
void spec_set_x(t_part *vector, int *np, const int range[][2], const int ppc[2],
		const t_density *part_density, const t_part_data dx[2], const int n_move)
{
	float *poscell;
	int start, end;

	// Calculate particle positions inside the cell
	const int npc = ppc[0] * ppc[1];
	t_part_data const dpcx = 1.0f / ppc[0];
	t_part_data const dpcy = 1.0f / ppc[1];

	poscell = malloc(2 * npc * sizeof(t_part_data));

	int ip = 0;
	for (int j = 0; j < ppc[1]; j++)
	{
		for (int i = 0; i < ppc[0]; i++)
		{
			poscell[ip] = dpcx * (i + 0.5);
			poscell[ip + 1] = dpcy * (j + 0.5);
			ip += 2;
		}
	}

	// Set position of particles in the specified grid range according to the density profile
	switch (part_density->type)
	{
		case STEP:    // Step like density profile

			// Get edge position normalized to cell size;
			start = part_density->start / dx[0] - n_move;
			if(range[0][0] > start) start = range[0][0];

			end = range[0][1];
			break;

		case SLAB:    // Slab like density profile
			// Get edge position normalized to cell size;
			start = part_density->start / dx[0] - n_move;
			end = part_density->end / dx[0] - n_move;

			if(start < range[0][0]) start = range[0][0];
			if(end > range[0][1]) end = range[0][1];
			break;

		default:    // Uniform density
			start = range[0][0];
			end = range[0][1];

	}

	ip = *np;

	for (int j = range[1][0]; j < range[1][1]; j++)
	{
		for (int i = start; i < end; i++)
		{
			for (int k = 0; k < npc; k++)
			{
				vector[ip].ix = i;
				vector[ip].iy = j;
				vector[ip].x = poscell[2 * k];
				vector[ip].y = poscell[2 * k + 1];
				ip++;
			}
		}
	}

	*np = ip;
	free(poscell);
}

// Inject the particles in the simulation
void spec_inject_particles(t_part **part, int *np, int *np_max, const int range[][2], const int ppc[2],
		const t_density *part_density, const t_part_data dx[2], const int n_move,
		const t_part_data ufl[3], const t_part_data uth[3])
{
	int start = *np;

	// Get maximum number of particles to inject
	int np_inj = (range[0][1] - range[0][0]) * (range[1][1] - range[1][0]) * ppc[0] * ppc[1];

	// Check if buffer is large enough and if not reallocate
	if (*np + np_inj > *np_max)
	{
		*np_max = ((*np_max + np_inj) / 1024 + 1) * 1024;
		*part = realloc((void*) *part, *np_max * sizeof(t_part));
	}

	// Set particle positions
	spec_set_x(*part, np, range, ppc, part_density, dx, n_move);

	// Set momentum of injected particles
	spec_set_u(*part, start, *np, ufl, uth);
}

void spec_new(t_species *spec, char name[], const t_part_data m_q, const int ppc[],
        const t_part_data *ufl, const t_part_data *uth, const int nx[], t_part_data box[],
        const float dt, t_density *density)
{

	int i, npc;

	// Species name
	strncpy(spec->name, name, MAX_SPNAME_LEN);

	npc = 1;
	// Store species data
	for (i = 0; i < 2; i++)
	{
		spec->nx[i] = nx[i];
		spec->ppc[i] = ppc[i];
		npc *= ppc[i];

		spec->box[i] = box[i];
		spec->dx[i] = box[i] / nx[i];
	}

	spec->m_q = m_q;
	spec->q = copysign(1.0f, m_q) / npc;

	spec->dt = dt;

	// Initialize particle buffer
	spec->np_max = 0;
	spec->part = NULL;

	// Initialize density profile
	if (density)
	{
		spec->density = *density;
		if (spec->density.n == 0.) spec->density.n = 1.0;
	} else
	{
		// Default values
		spec->density = (t_density ) { .type = UNIFORM, .n = 1.0 };
	}

	// Initialize temperature profile
	if (ufl)
	{
		for (i = 0; i < 3; i++)
			spec->ufl[i] = ufl[i];
	} else
	{
		for (i = 0; i < 3; i++)
			spec->ufl[i] = 0;
	}

	// Density multiplier
	spec->q *= fabsf(spec->density.n);

	if (uth)
	{
		for (i = 0; i < 3; i++)
			spec->uth[i] = uth[i];
	} else
	{
		for (i = 0; i < 3; i++)
			spec->uth[i] = 0;
	}

	// Reset iteration number
	spec->iter = 0;

	// Reset moving window information
	spec->moving_window = 0;
	spec->n_move = 0;

	// Inject initial particle distribution
	spec->np = 0;

	const int range[][2] = {{ 0, nx[0]}, { 0, nx[1]}};
	spec_inject_particles(&spec->part, &spec->np, &spec->np_max, range, spec->ppc, &spec->density,
	        spec->dx, spec->n_move, spec->ufl, spec->uth);

	spec->inj_part = NULL;
	spec->np_inj = 0;
}

void spec_move_window(t_species *spec)
{
	if ((spec->iter * spec->dt) > (spec->dx[0] * (spec->n_move + 1)))
	{
		// shift all particles left
		// particles leaving the box will be removed later
		int i;
		#pragma omp for
		for (i = 0; i < spec->np; i++)
			spec->part[i].ix--;

		#pragma omp single
		{
			// Increase moving window counter
			spec->n_move++;

			if(spec->np_inj == 0)
			{
				// Inject particles in the right edge of the simulation box
				const int range[][2] = { { spec->nx[0] - 1, spec->nx[0] }, { 0, spec->nx[1] } };
				int npc = spec->ppc[0] * spec->ppc[1];
				int max = npc * spec->nx[1];

				spec->inj_part = malloc(max * sizeof(t_part));
				spec->np_inj = 0;

				spec_inject_particles(&spec->inj_part, &spec->np_inj, &max, range, spec->ppc,
				        &spec->density, spec->dx, spec->n_move, spec->ufl, spec->uth);
			}

			if(spec->np_inj + spec->np > spec->np_max)
			{
				spec->np_max = ((spec->np_max + spec->np_inj) / 1024 + 1) * 1024;
				spec->part = realloc((void*) spec->part, spec->np_max * sizeof(t_part));
			}
		}

		int begin = spec->np;

		#pragma omp for
		for(i = 0; i < spec->np_inj; i++)
			spec->part[begin + i] = spec->inj_part[i];

		#pragma omp single
		spec->np += spec->np_inj;
	}
}

void spec_delete(t_species *spec)
{
	free(spec->part);
//	spec->np = -1;

	if (spec->np == 0) dealloc();
}

/*********************************************************************************************
 
 Cuurent deposition
 
 *********************************************************************************************/

void dep_current_esk(int ix0, int iy0, int di, int dj, t_part_data x0, t_part_data y0,
        t_part_data x1, t_part_data y1, t_part_data qvx, t_part_data qvy, t_part_data qvz,
        t_current *current)
{

	int i, j;
	t_fld S0x[4], S0y[4], S1x[4], S1y[4], DSx[4], DSy[4];
	t_fld Wx[16], Wy[16], Wz[16];

	S0x[0] = 0.0f;
	S0x[1] = 1.0f - x0;
	S0x[2] = x0;
	S0x[3] = 0.0f;

	S0y[0] = 0.0f;
	S0y[1] = 1.0f - y0;
	S0y[2] = y0;
	S0y[3] = 0.0f;

	for (i = 0; i < 4; i++)
	{
		S1x[i] = 0.0f;
		S1y[i] = 0.0f;
	}

	S1x[1 + di] = 1.0f - x1;
	S1x[2 + di] = x1;

	S1y[1 + dj] = 1.0f - y1;
	S1y[2 + dj] = y1;

	for (i = 0; i < 4; i++)
	{
		DSx[i] = S1x[i] - S0x[i];
		DSy[i] = S1y[i] - S0y[i];
	}

	for (j = 0; j < 4; j++)
	{
		for (i = 0; i < 4; i++)
		{
			Wx[i + 4 * j] = DSx[i] * (S0y[j] + DSy[j] / 2.0f);
			Wy[i + 4 * j] = DSy[j] * (S0x[i] + DSx[i] / 2.0f);
			Wz[i + 4 * j] = S0x[i] * S0y[j] + DSx[i] * S0y[j] / 2.0f + S0x[i] * DSy[j] / 2.0f
			        + DSx[i] * DSy[j] / 3.0f;
		}
	}

	// jx
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;

	for (j = 0; j < 4; j++)
	{
		t_fld c;

		c = -qvx * Wx[4 * j];
		J[ix0 - 1 + (iy0 - 1 + j) * nrow].x += c;
		for (i = 1; i < 4; i++)
		{
			c -= qvx * Wx[i + 4 * j];
			J[ix0 + i - 1 + (iy0 - 1 + j) * nrow].x += c;
		}
	}

	// jy
	for (i = 0; i < 4; i++)
	{
		t_fld c;

		c = -qvy * Wy[i];
		J[ix0 + i - 1 + (iy0 - 1) * nrow].y += c;
		for (j = 1; j < 4; j++)
		{
			c -= qvy * Wy[i + 4 * j];
			J[ix0 + i - 1 + (iy0 - 1 + j) * nrow].y += c;
		}
	}

	// jz
	for (j = 0; j < 4; j++)
	{
		for (i = 0; i < 4; i++)
		{
			J[ix0 + i - 1 + (iy0 - 1 + j) * nrow].z += qvz * Wz[i + 4 * j];
		}
	}

}

void dep_current_zamb(int ix, int iy, int di, int dj, float x0, float y0, float dx, float dy,
        float qnx, float qny, float qvz, t_current *current, t_vfld *J)
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
	int k;
	const int nrow = current->nrow;

	for (k = 0; k < vnp; k++)
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

		J[vp[k].ix + nrow * vp[k].iy].x += wl1 * wp1[0];
		J[vp[k].ix + nrow * (vp[k].iy + 1)].x += wl1 * wp1[1];

		J[vp[k].ix + nrow * vp[k].iy].y += wl2 * wp2[0];
		J[vp[k].ix + 1 + nrow * vp[k].iy].y += wl2 * wp2[1];

		J[vp[k].ix + nrow * vp[k].iy].z += vp[k].qvz
		        * (S0x[0] * S0y[0] + S1x[0] * S1y[0] + (S0x[0] * S1y[0] - S1x[0] * S0y[0]) / 2.0f);
		J[vp[k].ix + 1 + nrow * vp[k].iy].z += vp[k].qvz
		        * (S0x[1] * S0y[0] + S1x[1] * S1y[0] + (S0x[1] * S1y[0] - S1x[1] * S0y[0]) / 2.0f);
		J[vp[k].ix + nrow * (vp[k].iy + 1)].z += vp[k].qvz
		        * (S0x[0] * S0y[1] + S1x[0] * S1y[1] + (S0x[0] * S1y[1] - S1x[0] * S0y[1]) / 2.0f);
		J[vp[k].ix + 1 + nrow * (vp[k].iy + 1)].z += vp[k].qvz
		        * (S0x[1] * S0y[1] + S1x[1] * S1y[1] + (S0x[1] * S1y[1] - S1x[1] * S0y[1]) / 2.0f);
	}

}

/*********************************************************************************************
 
 Sorting
 
 *********************************************************************************************/

void spec_sort(t_species *spec)
{
	int *idx, *npic;

	int ncell = spec->nx[0] * spec->nx[1];

	// Allocate index memory
	idx = malloc(spec->np * sizeof(int));

	// Allocate temp. array with number of particles in cell
	npic = malloc(ncell * sizeof(int));
	memset(npic, 0, ncell * sizeof(int));

	// Generate sorted index
	int i;
	for (i = 0; i < spec->np; i++)
	{
		idx[i] = spec->part[i].ix + spec->part[i].iy * spec->nx[0];
		npic[idx[i]]++;
	}

	int isum = 0, j;
	for (i = 0; i < ncell; i++)
	{
		j = npic[i];
		npic[i] = isum;
		isum += j;
	}

	for (i = 0; i < spec->np; i++)
	{
		j = idx[i];
		idx[i] = npic[j]++;
	}

	// free temp. array
	free(npic);
	/*
	 // Rearrange particle buffer
	 t_part *tmp = malloc( spec->np * sizeof( t_part ) );
	 for (i=0; i< spec->np; i++) {
	 tmp[idx[i]] = spec->part[i];
	 }
	 free(spec->part);
	 spec->part = tmp;
	 */

	// low mem
	for (i = 0; i < spec->np; i++)
	{
		t_part tmp;
		int k;

		k = idx[i];
		while (k > i)
		{
			int t;

			tmp = spec->part[k];
			spec->part[k] = spec->part[i];
			spec->part[i] = tmp;

			t = idx[k];
			idx[k] = -1;
			k = t;
		}
	}

	free(idx);

}

/*********************************************************************************************
 
 Particle advance
 
 *********************************************************************************************/

void interpolate_fld(const t_vfld *restrict const E, const t_vfld *restrict const B, const int nrow,
        const t_part *restrict const part, t_vfld *restrict const Ep, t_vfld *restrict const Bp)
{
	register int i, j, ih, jh;
	register t_fld w1, w2, w1h, w2h;

	i = part->ix;
	j = part->iy;

	w1 = part->x;
	w2 = part->y;

	ih = (w1 < 0.5f) ? -1 : 0;
	jh = (w2 < 0.5f) ? -1 : 0;

	// w1h = w1 - 0.5f - ih;
	// w2h = w2 - 0.5f - jh;
	w1h = w1 + ((w1 < 0.5f) ? 0.5f : -0.5f);
	w2h = w2 + ((w2 < 0.5f) ? 0.5f : -0.5f);

	ih += i;
	jh += j;

	Ep->x = (E[ih + j * nrow].x * (1.0f - w1h) + E[ih + 1 + j * nrow].x * w1h) * (1.0f - w2)
	        + (E[ih + (j + 1) * nrow].x * (1.0f - w1h) + E[ih + 1 + (j + 1) * nrow].x * w1h) * w2;

	Ep->y = (E[i + jh * nrow].y * (1.0f - w1) + E[i + 1 + jh * nrow].y * w1) * (1.0f - w2h)
	        + (E[i + (jh + 1) * nrow].y * (1.0f - w1) + E[i + 1 + (jh + 1) * nrow].y * w1) * w2h;

	Ep->z = (E[i + j * nrow].z * (1.0f - w1) + E[i + 1 + j * nrow].z * w1) * (1.0f - w2)
	        + (E[i + (j + 1) * nrow].z * (1.0f - w1) + E[i + 1 + (j + 1) * nrow].z * w1) * w2;

	Bp->x = (B[i + jh * nrow].x * (1.0f - w1) + B[i + 1 + jh * nrow].x * w1) * (1.0f - w2h)
	        + (B[i + (jh + 1) * nrow].x * (1.0f - w1) + B[i + 1 + (jh + 1) * nrow].x * w1) * w2h;

	Bp->y = (B[ih + j * nrow].y * (1.0f - w1h) + B[ih + 1 + j * nrow].y * w1h) * (1.0f - w2)
	        + (B[ih + (j + 1) * nrow].y * (1.0f - w1h) + B[ih + 1 + (j + 1) * nrow].y * w1h) * w2;

	Bp->z = (B[ih + jh * nrow].z * (1.0f - w1h) + B[ih + 1 + jh * nrow].z * w1h) * (1.0f - w2h)
	        + (B[ih + (jh + 1) * nrow].z * (1.0f - w1h) + B[ih + 1 + (jh + 1) * nrow].z * w1h)
	                * w2h;

}

int ltrim(t_part_data x)
{
	return (x >= 1.0f) - (x < 0.0f);
}

void spec_advance(t_species *spec, t_emf *emf, t_current *current)
{
	int i;

	t_part_data qnx, qny, qvz;

	uint64_t t0;
	t0 = timer_ticks();

	const t_part_data tem = 0.5 * spec->dt / spec->m_q;
	const t_part_data dt_dx = spec->dt / spec->dx[0];
	const t_part_data dt_dy = spec->dt / spec->dx[1];

	// Auxiliary values for current deposition
	qnx = spec->q * spec->dx[0] / spec->dt;
	qny = spec->q * spec->dx[1] / spec->dt;

	const int nx0 = spec->nx[0];
	const int nx1 = spec->nx[1];

	reduce.J = current->J;
	reduce.J_buff_size = (current->gc[0][0] + current->nx[0] + current->gc[0][1])
	        * (current->gc[1][0] + current->nx[1] + current->gc[1][1]);
	/* J_size is equals to:
	 size of J_buf - the number of guard cells in the lower side */
	reduce.J_size = reduce.J_buff_size - (current->gc[0][0] + current->gc[1][0] * current->nrow);

	#pragma omp single
	if (reduce_array == NULL)
	{
		alloc(reduce.J_buff_size, reduce.J_size);
	}

	// Advance particles
	#pragma omp for private(qvz) reduction(tvfldAdd: reduce)
	for (i = 0; i < spec->np; i++)
	{
		t_vfld Ep, Bp;
		t_part_data utx, uty, utz;
		t_part_data ux, uy, uz, rg;
		t_part_data gtem, otsq;

		t_part_data x1, y1;

		int di, dj;
		float dx, dy;

		// Load particle momenta
		ux = spec->part[i].ux;
		uy = spec->part[i].uy;
		uz = spec->part[i].uz;

		// interpolate fields
		interpolate_fld(emf->E, emf->B, emf->nrow, &spec->part[i], &Ep, &Bp);

		// advance u using Boris scheme
		Ep.x *= tem;
		Ep.y *= tem;
		Ep.z *= tem;

		utx = ux + Ep.x;
		uty = uy + Ep.y;
		utz = uz + Ep.z;

		// Perform first half of the rotation
		gtem = tem / sqrtf(1.0f + utx * utx + uty * uty + utz * utz);

		Bp.x *= gtem;
		Bp.y *= gtem;
		Bp.z *= gtem;

		otsq = 2.0f / (1.0f + Bp.x * Bp.x + Bp.y * Bp.y + Bp.z * Bp.z);

		ux = utx + uty * Bp.z - utz * Bp.y;
		uy = uty + utz * Bp.x - utx * Bp.z;
		uz = utz + utx * Bp.y - uty * Bp.x;

		// Perform second half of the rotation
		Bp.x *= otsq;
		Bp.y *= otsq;
		Bp.z *= otsq;

		utx += uy * Bp.z - uz * Bp.y;
		uty += uz * Bp.x - ux * Bp.z;
		utz += ux * Bp.y - uy * Bp.x;

		// Perform second half of electric field acceleration
		ux = utx + Ep.x;
		uy = uty + Ep.y;
		uz = utz + Ep.z;

		// Store new momenta
		spec->part[i].ux = ux;
		spec->part[i].uy = uy;
		spec->part[i].uz = uz;

		// push particle
		rg = 1.0f / sqrtf(1.0f + ux * ux + uy * uy + uz * uz);

		dx = dt_dx * rg * ux;
		dy = dt_dy * rg * uy;

		x1 = spec->part[i].x + dx;
		y1 = spec->part[i].y + dy;

		di = ltrim(x1);
		dj = ltrim(y1);

		x1 -= di;
		y1 -= dj;

		qvz = spec->q * uz * rg;

		// deposit current using Eskirepov method
		// dep_current_esk( spec -> part[i].ix, spec -> part[i].iy, di, dj, 
		// 				 spec -> part[i].x, spec -> part[i].y, x1, y1, 
		// 				 qnx, qny, qvz, 
		// 				 current );

		dep_current_zamb(spec->part[i].ix, spec->part[i].iy, di, dj, spec->part[i].x,
		        spec->part[i].y, dx, dy, qnx, qny, qvz, current, reduce.J);

		// Store results
		spec->part[i].x = x1;
		spec->part[i].y = y1;
		spec->part[i].ix += di;
		spec->part[i].iy += dj;
	}

	#pragma omp single
	{
		// Advance internal iteration number
		spec->iter += 1;
		_spec_npush += spec->np;
	}

	// Check for particles leaving the box
	if (spec->moving_window)
	{

		// Move simulation window if needed
		spec_move_window(spec);

		// Use absorbing boundaries along x, periodic along y
// 		int id = omp_get_thread_num(), size = omp_get_num_threads(), aux;
// 		i = id;
// 		while ( i < spec -> np ) {
// 			if (( spec -> part[i].ix < 0 ) || ( spec -> part[i].ix >= nx0 )) {
// 				
// 				#pragma omp atomic capture
// 				aux = --spec -> np;
// 				
// 				spec -> part[i] = spec -> part[ aux ];
// 				continue; 
// 			}
// 			spec -> part[i].iy += (( spec -> part[i].iy < 0 ) ? nx1 : 0 ) - (( spec -> part[i].iy >= nx1 ) ? nx1 : 0);
// 			i += size;
// 		}

		int id = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		int aux;

		int begin = floor((float) id * spec->np / num_threads);
		int end = floor((float) (id + 1) * spec->np / num_threads);

		i = begin;
		while (i < end && i < spec->np)
		{
			if ((spec->part[i].ix < 0) || (spec->part[i].ix >= nx0))
			{
				#pragma omp atomic capture
				aux = --spec->np;

				spec->part[i] = spec->part[aux];
				continue;
			}
			spec->part[i].iy += ((spec->part[i].iy < 0) ? nx1 : 0)
			        - ((spec->part[i].iy >= nx1) ? nx1 : 0);
			i++;
		}
	} else
	{
		// Use periodic boundaries in both directions
		#pragma omp for
		for (i = 0; i < spec->np; i++)
		{
			spec->part[i].ix += ((spec->part[i].ix < 0) ? nx0 : 0)
			        - ((spec->part[i].ix >= nx0) ? nx0 : 0);
			spec->part[i].iy += ((spec->part[i].iy < 0) ? nx1 : 0)
			        - ((spec->part[i].iy >= nx1) ? nx1 : 0);
		}
	}

// 	#pragma omp single
// 	{
// 		Sort species at every 16 time steps
// 		if ( ! (spec -> iter % 16) ) spec_sort( spec );
// 
// 		_spec_time += timer_interval_seconds( t0, timer_ticks() );
// 	}
}

/*********************************************************************************************
 
 Charge Deposition
 
 *********************************************************************************************/

void spec_deposit_charge(const t_species *spec, t_part_data *charge)
{
	int i, j;

	// Charge array is expected to have 1 guard cell at the upper boundary
	int nrow = spec->nx[0] + 1;
	t_part_data q = spec->q;

	for (i = 0; i < spec->np; i++)
	{
		int idx = spec->part[i].ix + nrow * spec->part[i].iy;
		t_fld w1, w2;

		w1 = spec->part[i].x;
		w2 = spec->part[i].y;

		charge[idx] += (1.0f - w1) * (1.0f - w2) * q;
		charge[idx + 1] += (w1) * (1.0f - w2) * q;
		charge[idx + nrow] += (1.0f - w1) * (w2) * q;
		charge[idx + 1 + nrow] += (w1) * (w2) * q;
	}

	// Correct boundary values

	// x
	if (!spec->moving_window)
	{
		for (j = 0; j < spec->nx[1] + 1; j++)
		{
			charge[0 + j * nrow] += charge[spec->nx[0] + j * nrow];
		}
	}

	// y - Periodic boundaries
	for (i = 0; i < spec->nx[0] + 1; i++)
	{
		charge[i + 0] += charge[i + spec->nx[1] * nrow];
	}

}

/*********************************************************************************************
 
 Diagnostics
 
 *********************************************************************************************/

void spec_rep_particles(const t_species *spec)
{

	t_zdf_file part_file;

	int i;

	const char *quants[] = { "x1", "x2", "u1", "u2", "u3" };

	const char *units[] = { "c/\\omega_p", "c/\\omega_p", "c", "c", "c" };

	t_zdf_iteration iter = { .n = spec->iter, .t = spec->iter * spec->dt,
	        .time_units = "1/\\omega_p" };

	// Allocate buffer for positions

	t_zdf_part_info info = { .name = (char*) spec->name, .nquants = 5, .quants = (char**) quants,
	        .units = (char**) units, .np = spec->np };

	// Create file and add description
	zdf_part_file_open(&part_file, &info, &iter, "PARTICLES");

	// Add positions and generalized velocities
	size_t size = (spec->np) * sizeof(float);
	float *data = malloc(size);

	// x1
	for (i = 0; i < spec->np; i++)
		data[i] = (spec->n_move + spec->part[i].ix + spec->part[i].x) * spec->dx[0];
	zdf_part_file_add_quant(&part_file, quants[0], data, spec->np);

	// x2
	for (i = 0; i < spec->np; i++)
		data[i] = (spec->part[i].iy + spec->part[i].y) * spec->dx[1];
	zdf_part_file_add_quant(&part_file, quants[1], data, spec->np);

	// ux
	for (i = 0; i < spec->np; i++)
		data[i] = spec->part[i].ux;
	zdf_part_file_add_quant(&part_file, quants[2], data, spec->np);

	// uy
	for (i = 0; i < spec->np; i++)
		data[i] = spec->part[i].uy;
	zdf_part_file_add_quant(&part_file, quants[3], data, spec->np);

	// uz
	for (i = 0; i < spec->np; i++)
		data[i] = spec->part[i].uz;
	zdf_part_file_add_quant(&part_file, quants[4], data, spec->np);

	free(data);

	zdf_close_file(&part_file);
}

void spec_rep_charge(const t_species *spec)
{
	t_part_data *buf, *charge, *b, *c;
	size_t size;
	int i, j;

	// Add 1 guard cell to the upper boundary
	size = (spec->nx[0] + 1) * (spec->nx[1] + 1) * sizeof(t_part_data);
	charge = malloc(size);
	memset(charge, 0, size);

	// Deposit the charge
	spec_deposit_charge(spec, charge);

	// Compact the data to save the file (throw away guard cells)
	size = (spec->nx[0]) * (spec->nx[1]);
	buf = malloc(size * sizeof(float));

	b = buf;
	c = charge;
	for (j = 0; j < spec->nx[1]; j++)
	{
		for (i = 0; i < spec->nx[0]; i++)
		{
			b[i] = c[i];
		}
		b += spec->nx[0];
		c += spec->nx[0] + 1;
	}

	free(charge);

	t_zdf_grid_axis axis[2];
	axis[0] = (t_zdf_grid_axis ) { .min = 0.0, .max = spec->box[0], .label = "x_1",
			        .units = "c/\\omega_p" };

	axis[1] = (t_zdf_grid_axis ) { .min = 0.0, .max = spec->box[1], .label = "x_2",
			        .units = "c/\\omega_p" };

	t_zdf_grid_info info = { .ndims = 2, .label = "charge", .units = "n_e", .axis = axis };

	info.nx[0] = spec->nx[0];
	info.nx[1] = spec->nx[1];

	t_zdf_iteration iter = { .n = spec->iter, .t = spec->iter * spec->dt,
	        .time_units = "1/\\omega_p" };

	zdf_save_grid(buf, &info, &iter, spec->name);

	free(buf);
}

void spec_pha_axis(const t_species *spec, int i0, int np, int quant, float *axis)
{
	int i;

	switch (quant)
	{
		case X1:
			for (i = 0; i < np; i++)
				axis[i] = (spec->part[i0 + i].x + spec->part[i0 + i].ix) * spec->dx[0];
			break;
		case X2:
			for (i = 0; i < np; i++)
				axis[i] = (spec->part[i0 + i].y + spec->part[i0 + i].iy) * spec->dx[1];
			break;
		case U1:
			for (i = 0; i < np; i++)
				axis[i] = spec->part[i0 + i].ux;
			break;
		case U2:
			for (i = 0; i < np; i++)
				axis[i] = spec->part[i0 + i].uy;
			break;
		case U3:
			for (i = 0; i < np; i++)
				axis[i] = spec->part[i0 + i].uz;
			break;
	}
}

const char* spec_pha_axis_units(int quant)
{
	switch (quant)
	{
		case X1:
		case X2:
			return ("c/\\omega_p");
			break;
		case U1:
		case U2:
		case U3:
			return ("m_e c");
	}
	return ("");
}

void spec_deposit_pha(const t_species *spec, const int rep_type, const int pha_nx[],
        const float pha_range[][2], float *restrict buf)
{
	const int BUF_SIZE = 1024;
	float pha_x1[BUF_SIZE], pha_x2[BUF_SIZE];

	const int nrow = pha_nx[0];

	const int quant1 = rep_type & 0x000F;
	const int quant2 = (rep_type & 0x00F0) >> 4;

	const float x1min = pha_range[0][0];
	const float x2min = pha_range[1][0];

	const float rdx1 = pha_nx[0] / (pha_range[0][1] - pha_range[0][0]);
	const float rdx2 = pha_nx[1] / (pha_range[1][1] - pha_range[1][0]);

	for (int i = 0; i < spec->np; i += BUF_SIZE)
	{
		int np = (i + BUF_SIZE > spec->np) ? spec->np - i : BUF_SIZE;

		spec_pha_axis(spec, i, np, quant1, pha_x1);
		spec_pha_axis(spec, i, np, quant2, pha_x2);

		for (int k = 0; k < np; k++)
		{

			float nx1 = (pha_x1[k] - x1min) * rdx1;
			float nx2 = (pha_x2[k] - x2min) * rdx2;

			int i1 = (int) (nx1 + 0.5f);
			int i2 = (int) (nx2 + 0.5f);

			float w1 = nx1 - i1 + 0.5f;
			float w2 = nx2 - i2 + 0.5f;

			int idx = i1 + nrow * i2;

			if (i2 >= 0 && i2 < pha_nx[1])
			{

				if (i1 >= 0 && i1 < pha_nx[0])
				{
					buf[idx] += (1.0f - w1) * (1.0f - w2) * spec->q;
				}

				if (i1 + 1 >= 0 && i1 + 1 < pha_nx[0])
				{
					buf[idx + 1] += w1 * (1.0f - w2) * spec->q;
				}
			}

			idx += nrow;
			if (i2 + 1 >= 0 && i2 + 1 < pha_nx[1])
			{

				if (i1 >= 0 && i1 < pha_nx[0])
				{
					buf[idx] += (1.0f - w1) * w2 * spec->q;
				}

				if (i1 + 1 >= 0 && i1 + 1 < pha_nx[0])
				{
					buf[idx + 1] += w1 * w2 * spec->q;
				}
			}

		}

	}
}

void spec_rep_pha(const t_species *spec, const int rep_type, const int pha_nx[],
        const float pha_range[][2])
{

	char const *const pha_ax_name[] = { "x1", "x2", "x3", "u1", "u2", "u3" };
	char pha_name[64];

	// Allocate phasespace buffer
	float *restrict buf = malloc(pha_nx[0] * pha_nx[1] * sizeof(float));
	memset(buf, 0, pha_nx[0] * pha_nx[1] * sizeof(float));

	// Deposit the phasespace
	spec_deposit_pha(spec, rep_type, pha_nx, pha_range, buf);

	// save the data in hdf5 format
	int quant1 = rep_type & 0x000F;
	int quant2 = (rep_type & 0x00F0) >> 4;

	const char *pha_ax1_units = spec_pha_axis_units(quant1);
	const char *pha_ax2_units = spec_pha_axis_units(quant2);

	sprintf(pha_name, "%s%s", pha_ax_name[quant1 - 1], pha_ax_name[quant2 - 1]);

	t_zdf_grid_axis axis[2];
	axis[0] = (t_zdf_grid_axis ) { .min = pha_range[0][0], .max = pha_range[0][1],
			        .label = (char*) pha_ax_name[quant1 - 1], .units = (char*) pha_ax1_units };

	axis[1] = (t_zdf_grid_axis ) { .min = pha_range[1][0], .max = pha_range[1][1],
			        .label = (char*) pha_ax_name[quant2 - 1], .units = (char*) pha_ax2_units };

	t_zdf_grid_info info = { .ndims = 2, .label = pha_name, .units = "a.u.", .axis = axis };

	info.nx[0] = pha_nx[0];
	info.nx[1] = pha_nx[1];

	t_zdf_iteration iter = { .n = spec->iter, .t = spec->iter * spec->dt,
	        .time_units = "1/\\omega_p" };

	zdf_save_grid(buf, &info, &iter, spec->name);

	// Free temp. buffer
	free(buf);

}

void spec_report(const t_species *spec, const int rep_type, const int pha_nx[],
        const float pha_range[][2])
{

	switch (rep_type & 0xF000)
	{
		case CHARGE:
			spec_rep_charge(spec);
			break;

		case PHA:
			spec_rep_pha(spec, rep_type, pha_nx, pha_range);
			break;

		case PARTICLES:
			spec_rep_particles(spec);
			break;
	}

}
