/*********************************************************************************************
 ZPIC
 particles.c

 Created by Ricardo Fonseca on 11/8/10.
 Modified by Nicolas Guidotti on 11/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#include "particles.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <cuda.h>
#include <vector_types.h>

#include "random.h"
#include "utilities.h"
#include "zdf.h"

typedef struct {
	float x0, x1, y0, y1, dx, dy, qvz;
	int ix, iy;
} t_vp;

/*********************************************************************************************
 * Vector Utilities
 *********************************************************************************************/

void part_vector_alloc(t_part_vector *vector, const int64_t size_max, const int device)
{

	vector->ix = alloc_device_buffer(size_max * sizeof(int), device);
	vector->iy = alloc_device_buffer(size_max * sizeof(int), device);
	vector->x = alloc_device_buffer(size_max * sizeof(t_part_data), device);
	vector->y = alloc_device_buffer(size_max * sizeof(t_part_data), device);
	vector->ux = alloc_device_buffer(size_max * sizeof(t_part_data), device);
	vector->uy = alloc_device_buffer(size_max * sizeof(t_part_data), device);
	vector->uz = alloc_device_buffer(size_max * sizeof(t_part_data), device);
	vector->invalid = alloc_device_buffer(size_max * sizeof(bool), device);

	vector->size_max = size_max;
	vector->size = 0;
	vector->enable_vector = true;
}

void part_vector_free(t_part_vector *vector)
{

	free_device_buffer(vector->ix);
	free_device_buffer(vector->iy);
	free_device_buffer(vector->x);
	free_device_buffer(vector->y);
	free_device_buffer(vector->ux);
	free_device_buffer(vector->uy);
	free_device_buffer(vector->uz);
	free_device_buffer(vector->invalid);
}

void part_vector_realloc(t_part_vector *vector, const int64_t new_size, const int device)
{
	vector->size_max = new_size;

	realloc_device_buffer((void**) &vector->ix, vector->size, vector->size_max, sizeof(int), device);
	realloc_device_buffer((void**) &vector->iy, vector->size, vector->size_max, sizeof(int), device);
	realloc_device_buffer((void**) &vector->x, vector->size, vector->size_max, sizeof(t_part_data), device);
	realloc_device_buffer((void**) &vector->y, vector->size, vector->size_max, sizeof(t_part_data), device);
	realloc_device_buffer((void**) &vector->ux, vector->size, vector->size_max, sizeof(t_part_data), device);
	realloc_device_buffer((void**) &vector->uy, vector->size, vector->size_max, sizeof(t_part_data), device);
	realloc_device_buffer((void**) &vector->uz, vector->size, vector->size_max, sizeof(t_part_data), device);
	realloc_device_buffer((void**) &vector->invalid, vector->size, vector->size_max, sizeof(bool), device);
}

void part_vector_assign_valid_part(const t_part_vector *source, const int64_t source_idx,
								   t_part_vector *target, const int64_t target_idx)
{
	target->ix[target_idx] = source->ix[source_idx];
	target->iy[target_idx] = source->iy[source_idx];
	target->x[target_idx] = source->x[source_idx];
	target->y[target_idx] = source->y[source_idx];
	target->ux[target_idx] = source->ux[source_idx];
	target->uy[target_idx] = source->uy[source_idx];
	target->uz[target_idx] = source->uz[source_idx];
	target->invalid[target_idx] = source->invalid[source_idx];
}

void part_vector_memcpy(const t_part_vector *source, t_part_vector *target, const int64_t begin,
						const int64_t size)
{
	memcpy(target->ix, source->ix + begin, size * sizeof(int));
	memcpy(target->iy, source->iy + begin, size * sizeof(int));
	memcpy(target->x, source->x + begin, size * sizeof(t_part_data));
	memcpy(target->y, source->y + begin, size * sizeof(t_part_data));
	memcpy(target->ux, source->ux + begin, size * sizeof(t_part_data));
	memcpy(target->uy, source->uy + begin, size * sizeof(t_part_data));
	memcpy(target->uz, source->uz + begin, size * sizeof(t_part_data));
	memcpy(target->invalid, source->invalid + begin, size * sizeof(bool));
}

void part_vector_mem_advise(t_part_vector *vector, const int advise, const int device)
{
	cuMemAdvise(vector->ix, vector->size_max * sizeof(int), advise, device);
	cuMemAdvise(vector->iy, vector->size_max * sizeof(int), advise, device);
	cuMemAdvise(vector->x, vector->size_max * sizeof(t_part_data), advise, device);
	cuMemAdvise(vector->y, vector->size_max * sizeof(t_part_data), advise, device);
	cuMemAdvise(vector->ux, vector->size_max * sizeof(t_part_data), advise, device);
	cuMemAdvise(vector->uy, vector->size_max * sizeof(t_part_data), advise, device);
	cuMemAdvise(vector->uz, vector->size_max * sizeof(t_part_data), advise, device);
	cuMemAdvise(vector->invalid, vector->size_max * sizeof(bool), advise, device);
}

// Prefetching for particle vector
#ifdef ENABLE_PREFETCH
void spec_prefetch_openacc(t_part_vector *part, const int device, void *stream)
{
	cuMemPrefetchAsync(part->ix, part->size_max * sizeof(int), device, stream);
	cuMemPrefetchAsync(part->iy, part->size_max * sizeof(int), device, stream);
	cuMemPrefetchAsync(part->x, part->size_max * sizeof(t_part_data), device, stream);
	cuMemPrefetchAsync(part->y, part->size_max * sizeof(t_part_data), device, stream);
	cuMemPrefetchAsync(part->ux, part->size_max * sizeof(t_part_data), device, stream);
	cuMemPrefetchAsync(part->uy, part->size_max * sizeof(t_part_data), device, stream);
	cuMemPrefetchAsync(part->uz, part->size_max * sizeof(t_part_data), device, stream);
	cuMemPrefetchAsync(part->invalid, part->size_max * sizeof(bool), device, stream);
}
#endif

/*********************************************************************************************
 Initialization
 *********************************************************************************************/

// Set the momentum of the injected particles
void spec_set_u(t_part_vector *vector, const int64_t start, const int64_t end, const t_part_data ufl[3],
		const t_part_data uth[3])
{
	for (int64_t i = start; i < end; i++)
	{
		vector->ux[i] = ufl[0] + uth[0] * rand_norm();
		vector->uy[i] = ufl[1] + uth[1] * rand_norm();
		vector->uz[i] = ufl[2] + uth[2] * rand_norm();
	}
}

// Set the position of the injected particles
void spec_set_x(t_part_vector *vector, const int range[][2], const int ppc[2],
		const t_density *part_density, const t_part_data dx[2], const int n_move)
{
	int start, end;

	// Calculate particle positions inside the cell
	const int npc = ppc[0] * ppc[1];
	t_part_data const dpcx = 1.0f / ppc[0];
	t_part_data const dpcy = 1.0f / ppc[1];

	float *poscell = malloc(2 * npc * sizeof(t_part_data));

	int64_t ip = 0;
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

	ip = vector->size;

	// Set particle position and cell index
	for (int j = range[1][0]; j < range[1][1]; j++)
	{
		for (int i = start; i < end; i++)
		{
			for (int k = 0; k < npc; k++)
			{
				vector->ix[ip] = i;
				vector->iy[ip] = j;
				vector->x[ip] = poscell[2 * k];
				vector->y[ip] = poscell[2 * k + 1];
				vector->invalid[ip] = false;
				ip++;
			}
		}
	}

	vector->size = ip;
	free(poscell);
}

// Inject the particles in the simulation
void spec_inject_particles(t_part_vector *part_vector, const int range[][2], const int ppc[2],
		const t_density *part_density, const t_part_data dx[2], const int n_move,
		const t_part_data ufl[3], const t_part_data uth[3])
{
	int64_t start = part_vector->size;

	// Get maximum number of particles to inject
	int64_t np_inj = (range[0][1] - range[0][0]) * (range[1][1] - range[1][0]) * ppc[0] * ppc[1];

	// Check if buffer is large enough and if not reallocate
	if (start + np_inj > part_vector->size_max)
		part_vector_realloc(part_vector, ((part_vector->size_max + np_inj) / 1024 + 1) * 1024, 0);

	// Set particle positions
	spec_set_x(part_vector, range, ppc, part_density, dx, n_move);

	// Set momentum of injected particles
	spec_set_u(part_vector, start, part_vector->size, ufl, uth);
}

// Apply the sorting to one of the particle vectors (full version)
void spec_apply_full_sort(uint32_t *restrict vector, const int64_t *restrict target_idx, const int64_t move_size)
{
	uint32_t *restrict temp = malloc(move_size * sizeof(uint32_t));

	#pragma omp parallel for
	for (int64_t i = 0; i < move_size; i++)
		temp[i] = vector[i];

	#pragma omp parallel for
	for (int64_t i = 0; i < move_size; i++)
		if (target_idx[i] >= 0) vector[target_idx[i]] = temp[i];

	free(temp);
}

// Organize the particles in tiles (Bucket Sort)
void spec_organize_in_tiles(t_species *spec, const int limits_y[2], const int device)
{
	const int64_t size = spec->main_vector.size;
	const int n_tiles_x = spec->n_tiles_x;
	const int n_tiles_y = spec->n_tiles_y;

	spec->mv_part_offset = alloc_device_buffer((n_tiles_y * n_tiles_x + 1) * sizeof(int64_t), device);
	spec->tile_offset = alloc_device_buffer((n_tiles_y * n_tiles_x + 1) * sizeof(int64_t), device);
	int64_t *restrict tile_offset = spec->tile_offset;
	int64_t *restrict pos = malloc(size * sizeof(int64_t));

	memset(spec->tile_offset, 0, (n_tiles_y * n_tiles_x + 1) * sizeof(int64_t));
	memset(spec->mv_part_offset, 0, (n_tiles_y * n_tiles_x + 1) * sizeof(int64_t));

	// Calculate the histogram (number of particles per tile)
	#pragma omp parallel for
	for (int64_t i = 0; i < size; i++)
	{
		int ix = spec->main_vector.ix[i] / TILE_SIZE;
		int iy = (spec->main_vector.iy[i] - limits_y[0]) / TILE_SIZE;

		#pragma omp atomic capture
		pos[i] = tile_offset[ix + iy * n_tiles_x]++;
	}

	// Prefix sum to find the initial idx of each tile in the particle vector
	prefix_sum_serial(tile_offset, n_tiles_x * n_tiles_y + 1);

	// Calculate the target position of each particle
	#pragma omp parallel for
	for (int64_t i = 0; i < size; i++)
	{
		int ix = spec->main_vector.ix[i] / TILE_SIZE;
		int iy = (spec->main_vector.iy[i] - limits_y[0]) / TILE_SIZE;

		pos[i] += tile_offset[ix + iy * n_tiles_x];
	}

	const int64_t final_size = tile_offset[n_tiles_x * n_tiles_y];
	spec->main_vector.size = final_size;

	// Organize the particles in tiles based on the position vector
	spec_apply_full_sort((uint32_t*) spec->main_vector.ix, pos, size);
	spec_apply_full_sort((uint32_t*) spec->main_vector.iy, pos, size);
	spec_apply_full_sort((uint32_t*) spec->main_vector.x, pos, size);
	spec_apply_full_sort((uint32_t*) spec->main_vector.y, pos, size);
	spec_apply_full_sort((uint32_t*) spec->main_vector.ux, pos, size);
	spec_apply_full_sort((uint32_t*) spec->main_vector.uy, pos, size);
	spec_apply_full_sort((uint32_t*) spec->main_vector.uz, pos, size);

	// Validate all the particles
	#pragma omp parallel for
	for (int k = 0; k < final_size; k++)
		spec->main_vector.invalid[k] = false;

	free(pos);  // Clean position vector
}

// Constructor
void spec_new(t_species *spec, char name[], const t_part_data m_q, const int ppc[],
		const t_part_data *ufl, const t_part_data *uth, const int nx[], t_part_data box[],
		const float dt, t_density *density, const int region_size, const int device)
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
	spec->energy = 0;

	const int64_t np = (int64_t)((1 + EXTRA_NP) * npc * region_size * nx[0] / 1024UL + 1UL) * 1024UL;

	// Initialize particle buffer
	part_vector_alloc(&spec->main_vector, np, device);

	// Initialize temp buffer
	for (i = 0; i < 2; i++)
		part_vector_alloc(&spec->incoming_part[i], (npc * nx[0] / 1024UL + 1UL) * 1024UL, device);

	spec->incoming_part[2].enable_vector = false;

	// Initialize density profile
	if (density)
	{
		spec->density = *density;
		if (spec->density.n == 0.) spec->density.n = 1.0;
	} else
	{
		// Default values
		spec->density = (t_density) {.type = UNIFORM, .n = 1.0};
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
	spec->moving_window = false;
	spec->n_move = 0;

	// Sort
	spec->n_tiles_x = ceil((float) spec->nx[0] / TILE_SIZE);
	spec->n_tiles_y = ceil((float) region_size / TILE_SIZE);
	spec->tile_offset = NULL;
	spec->mv_part_offset = NULL;
}

void spec_delete(t_species *spec)
{
	part_vector_free(&spec->main_vector);

	if (spec->tile_offset)
		free_device_buffer(spec->tile_offset);
	if(spec->mv_part_offset)
		free_device_buffer(spec->mv_part_offset);

	for(int n = 0; n < 3; n++)
		if(spec->incoming_part[n].enable_vector)
			part_vector_free(&spec->incoming_part[n]);
}

/*********************************************************************************************
 Current deposition
 *********************************************************************************************/
// Current deposition (adapted Villasenor-Bunemann method). OpenAcc
//#pragma acc routine
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
	t_part_data S0x[2], S1x[2], S0y[2], S1y[2];
	t_part_data wl1, wl2;
	t_part_data wp1[2], wp2[2];

	const int idx = vp[begin].ix + nrow * vp[begin].iy;

	S0x[0] = 1.0f - vp[begin].x0;
	S0x[1] = vp[begin].x0;

	S1x[0] = 1.0f - vp[begin].x1;
	S1x[1] = vp[begin].x1;

	S0y[0] = 1.0f - vp[begin].y0;
	S0y[1] = vp[begin].y0;

	S1y[0] = 1.0f - vp[begin].y1;
	S1y[1] = vp[begin].y1;

	wl1 = qnx * vp[begin].dx;
	wl2 = qny * vp[begin].dy;

	wp1[0] = 0.5f * (S0y[0] + S1y[0]);
	wp1[1] = 0.5f * (S0y[1] + S1y[1]);

	wp2[0] = 0.5f * (S0x[0] + S1x[0]);
	wp2[1] = 0.5f * (S0x[1] + S1x[1]);

	#pragma acc atomic
	J[idx].x += wl1 * wp1[0];

	#pragma acc atomic
	J[idx].y += wl2 * wp2[0];

	#pragma acc atomic
	J[idx].z += vp[begin].qvz * (S0x[0] * S0y[0] + S1x[0] * S1y[0]
	             + (S0x[0] * S1y[0] - S1x[0] * S0y[0]) / 2.0f);

	#pragma acc atomic
	J[idx + 1].y += wl2 * wp2[1];

	#pragma acc atomic
	J[idx + 1].z += vp[begin].qvz * (S0x[1] * S0y[0] + S1x[1] * S1y[0]
	                + (S0x[1] * S1y[0] - S1x[1] * S0y[0]) / 2.0f);

	#pragma acc atomic
	J[idx + nrow].x += wl1 * wp1[1];

	#pragma acc atomic
	J[idx + nrow].z += vp[begin].qvz * (S0x[0] * S0y[1] + S1x[0] * S1y[1]
	                   + (S0x[0] * S1y[1] - S1x[0] * S0y[1]) / 2.0f);

	#pragma acc atomic
	J[idx + 1 + nrow].z += vp[begin].qvz * (S0x[1] * S0y[1] + S1x[1] * S1y[1]
	                       + (S0x[1] * S1y[1] - S1x[1] * S0y[1]) / 2.0f);

	if(vnp > 1)
	{
		S0x[0] = 1.0f - vp[begin + 1].x0;
		S0x[1] = vp[begin + 1].x0;

		S1x[0] = 1.0f - vp[begin + 1].x1;
		S1x[1] = vp[begin + 1].x1;

		S0y[0] = 1.0f - vp[begin + 1].y0;
		S0y[1] = vp[begin + 1].y0;

		S1y[0] = 1.0f - vp[begin + 1].y1;
		S1y[1] = vp[begin + 1].y1;

		wl1 = qnx * vp[begin + 1].dx;
		wl2 = qny * vp[begin + 1].dy;

		wp1[0] = 0.5f * (S0y[0] + S1y[0]);
		wp1[1] = 0.5f * (S0y[1] + S1y[1]);

		wp2[0] = 0.5f * (S0x[0] + S1x[0]);
		wp2[1] = 0.5f * (S0x[1] + S1x[1]);

		#pragma acc atomic
		J[vp[begin + 1].ix + nrow * vp[begin + 1].iy].x += wl1 * wp1[0];

		#pragma acc atomic
		J[vp[begin + 1].ix + nrow * (vp[begin + 1].iy + 1)].x += wl1 * wp1[1];

		#pragma acc atomic
		J[vp[begin + 1].ix + nrow * vp[begin + 1].iy].y += wl2 * wp2[0];

		#pragma acc atomic
		J[vp[begin + 1].ix + 1 + nrow * vp[begin + 1].iy].y += wl2 * wp2[1];

		#pragma acc atomic
		J[vp[begin + 1].ix + nrow * vp[begin + 1].iy].z += vp[begin + 1].qvz
		        * (S0x[0] * S0y[0] + S1x[0] * S1y[0] + (S0x[0] * S1y[0] - S1x[0] * S0y[0]) / 2.0f);

		#pragma acc atomic
		J[vp[begin + 1].ix + 1 + nrow * vp[begin + 1].iy].z += vp[begin + 1].qvz
		        * (S0x[1] * S0y[0] + S1x[1] * S1y[0] + (S0x[1] * S1y[0] - S1x[1] * S0y[0]) / 2.0f);

		#pragma acc atomic
		J[vp[begin + 1].ix + nrow * (vp[begin + 1].iy + 1)].z += vp[begin + 1].qvz
		        * (S0x[0] * S0y[1] + S1x[0] * S1y[1] + (S0x[0] * S1y[1] - S1x[0] * S0y[1]) / 2.0f);

		#pragma acc atomic
		J[vp[begin + 1].ix + 1 + nrow * (vp[begin + 1].iy + 1)].z += vp[begin + 1].qvz
		        * (S0x[1] * S0y[1] + S1x[1] * S1y[1] + (S0x[1] * S1y[1] - S1x[1] * S0y[1]) / 2.0f);

		if(vnp == 3)
		{
			S0x[0] = 1.0f - vp[begin + 2].x0;
			S0x[1] = vp[begin + 2].x0;

			S1x[0] = 1.0f - vp[begin + 2].x1;
			S1x[1] = vp[begin + 2].x1;

			S0y[0] = 1.0f - vp[begin + 2].y0;
			S0y[1] = vp[begin + 2].y0;

			S1y[0] = 1.0f - vp[begin + 2].y1;
			S1y[1] = vp[begin + 2].y1;

			wl1 = qnx * vp[begin + 2].dx;
			wl2 = qny * vp[begin + 2].dy;

			wp1[0] = 0.5f * (S0y[0] + S1y[0]);
			wp1[1] = 0.5f * (S0y[1] + S1y[1]);

			wp2[0] = 0.5f * (S0x[0] + S1x[0]);
			wp2[1] = 0.5f * (S0x[1] + S1x[1]);

			#pragma acc atomic
			J[vp[begin + 2].ix + nrow * vp[begin + 2].iy].x += wl1 * wp1[0];

			#pragma acc atomic
			J[vp[begin + 2].ix + nrow * (vp[begin + 2].iy + 1)].x += wl1 * wp1[1];

			#pragma acc atomic
			J[vp[begin + 2].ix + nrow * vp[begin + 2].iy].y += wl2 * wp2[0];

			#pragma acc atomic
			J[vp[begin + 2].ix + 1 + nrow * vp[begin + 2].iy].y += wl2 * wp2[1];

			#pragma acc atomic
			J[vp[begin + 2].ix + nrow * vp[begin + 2].iy].z += vp[begin + 2].qvz
			        * (S0x[0] * S0y[0] + S1x[0] * S1y[0] + (S0x[0] * S1y[0] - S1x[0] * S0y[0]) / 2.0f);

			#pragma acc atomic
			J[vp[begin + 2].ix + 1 + nrow * vp[begin + 2].iy].z += vp[begin + 2].qvz
			        * (S0x[1] * S0y[0] + S1x[1] * S1y[0] + (S0x[1] * S1y[0] - S1x[1] * S0y[0]) / 2.0f);

			#pragma acc atomic
			J[vp[begin + 2].ix + nrow * (vp[begin + 2].iy + 1)].z += vp[begin + 2].qvz
			        * (S0x[0] * S0y[1] + S1x[0] * S1y[1] + (S0x[0] * S1y[1] - S1x[0] * S0y[1]) / 2.0f);

			#pragma acc atomic
			J[vp[begin + 2].ix + 1 + nrow * (vp[begin + 2].iy + 1)].z += vp[begin + 2].qvz
			        * (S0x[1] * S0y[1] + S1x[1] * S1y[1] + (S0x[1] * S1y[1] - S1x[1] * S0y[1]) / 2.0f);
		}
	}
}

/*********************************************************************************************
 Particle advance
 *********************************************************************************************/
// EM fields interpolation. OpenAcc
//#pragma acc routine
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

// Advance u using Boris scheme. OpenAcc
//#pragma acc routine
void advance_part_momentum(float3 *part_velocity, t_vfld Ep, t_vfld Bp, const t_part_data tem)
{
	Ep.x *= tem;
	Ep.y *= tem;
	Ep.z *= tem;

	float3 ut;
	ut.x = part_velocity->x + Ep.x;
	ut.y = part_velocity->y + Ep.y;
	ut.z = part_velocity->z + Ep.z;

	// Perform first half of the rotation
	t_part_data ustq = ut.x * ut.x + ut.y * ut.y + ut.z * ut.z;
	t_part_data gtem = tem / sqrtf(1.0f + ustq);

	Bp.x *= gtem;
	Bp.y *= gtem;
	Bp.z *= gtem;

	part_velocity->x = ut.x + ut.y * Bp.z - ut.z * Bp.y;
	part_velocity->y = ut.y + ut.z * Bp.x - ut.x * Bp.z;
	part_velocity->z = ut.z + ut.x * Bp.y - ut.y * Bp.x;

	// Perform second half of the rotation
	t_part_data Bp_mag = Bp.x * Bp.x + Bp.y * Bp.y + Bp.z * Bp.z;
	t_part_data otsq = 2.0f / (1.0f + Bp_mag);

	Bp.x *= otsq;
	Bp.y *= otsq;
	Bp.z *= otsq;

	ut.x += part_velocity->y * Bp.z - part_velocity->z * Bp.y;
	ut.y += part_velocity->z * Bp.x - part_velocity->x * Bp.z;
	ut.z += part_velocity->x * Bp.y - part_velocity->y * Bp.x;

	// Perform second half of electric field acceleration
	part_velocity->x = ut.x + Ep.x;
	part_velocity->y = ut.y + Ep.y;
	part_velocity->z = ut.z + Ep.z;
}

// Particle advance (OpenAcc). Optimised for GPU architecture
void spec_advance_openacc(t_species *restrict const spec, const t_emf *restrict const emf,
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

	spec->npush += spec->main_vector.size;

	// Advance particles
	#pragma acc parallel loop gang collapse(2) vector_length(THREAD_BLOCK)
	for(int tile_y = 0; tile_y < spec->n_tiles_y; tile_y++)
	{
		for(int tile_x = 0; tile_x < spec->n_tiles_x; tile_x++)
		{
			const int tile_idx = tile_x + tile_y * spec->n_tiles_x;
			const int64_t begin = spec->tile_offset[tile_idx];
			const int64_t end = spec->tile_offset[tile_idx + 1];

			// Where the tile begin (lower left)
			int2 begin_idx;
			begin_idx.x = tile_x * TILE_SIZE;
			begin_idx.y = tile_y * TILE_SIZE;

			// Store local E, B and J (of a given tile) in the Shared Memory
			t_vfld E[(TILE_SIZE + 2) * (TILE_SIZE + 2)];
			t_vfld B[(TILE_SIZE + 2) * (TILE_SIZE + 2)];
			t_vfld J[(TILE_SIZE + 3) * (TILE_SIZE + 3)];

			#pragma acc cache(E[0 : (TILE_SIZE + 2) * (TILE_SIZE + 2)])
			#pragma acc cache(B[0 : (TILE_SIZE + 2) * (TILE_SIZE + 2)])
			#pragma acc cache(J[0 : (TILE_SIZE + 3) * (TILE_SIZE + 3)])

			t_vp vp[THREAD_BLOCK * 3];
			#pragma acc cache(vp[0 : THREAD_BLOCK * 3])

			// Load the EMF into the shared memory
			#pragma acc loop vector collapse(2)
			for(int j = 0; j < (TILE_SIZE + 2); j++)
			{
				for(int i = 0; i < (TILE_SIZE + 2); i++)
				{
					int2 idx;
					idx.x = begin_idx.x + i;
					idx.y = begin_idx.y + j;

					if(idx.x <= spec->nx[0] + 1 && idx.y <= (limits_y[1] - region_offset) + 1)
					{
						E[i + j * (TILE_SIZE + 2)] = emf->E_buf[idx.x + idx.y * nrow];
						B[i + j * (TILE_SIZE + 2)] = emf->B_buf[idx.x + idx.y * nrow];
					}else
					{
						E[i + j * (TILE_SIZE + 2)].x = 0;
						E[i + j * (TILE_SIZE + 2)].y = 0;
						E[i + j * (TILE_SIZE + 2)].z = 0;

						B[i + j * (TILE_SIZE + 2)].x = 0;
						B[i + j * (TILE_SIZE + 2)].y = 0;
						B[i + j * (TILE_SIZE + 2)].z = 0;
					}
				}
			}

			// Reset the local current to 0
			#pragma acc loop vector
			for(int i = 0; i < (TILE_SIZE + 3) * (TILE_SIZE + 3); i++)
			{
				J[i].x = 0.0f;
				J[i].y = 0.0f;
				J[i].z = 0.0f;
			}

			#pragma acc loop vector
			for (int64_t k = begin; k < end; k++)
			{
				// Load particle momenta into local variable
				float3 part_velocity;
				part_velocity.x = spec->main_vector.ux[k];
				part_velocity.y = spec->main_vector.uy[k];
				part_velocity.z = spec->main_vector.uz[k];

				// Load particle position into local variable
				float2 part_pos;
				part_pos.x = spec->main_vector.x[k];
				part_pos.y = spec->main_vector.y[k];

				// Load particle cell index into local variable
				// Then convert to local coordinates
				int2 part_idx;
				part_idx.x = spec->main_vector.ix[k] - (begin_idx.x - 1);
				part_idx.y = spec->main_vector.iy[k] - (begin_idx.y - 1) - region_offset;

				t_vfld Ep, Bp;

				// Interpolate fields
				interpolate_fld_openacc(E, B, (TILE_SIZE + 2), part_idx.x, part_idx.y, part_pos.x,
				        part_pos.y, &Ep, &Bp);

				// Advance the particle momentum
				advance_part_momentum(&part_velocity, Ep, Bp, tem);

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

				// Deposit current
				dep_current_openacc(part_idx.x, part_idx.y, di, dj, part_pos.x, part_pos.y, dx, dy,
				        qnx, qny, qvz, J, (TILE_SIZE + 3), vp, k % THREAD_BLOCK);

				// Store results
				spec->main_vector.x[k] = x1 - di;
				spec->main_vector.y[k] = y1 - dj;
				spec->main_vector.ix[k] += di;
				spec->main_vector.iy[k] += dj;
				spec->main_vector.ux[k] = part_velocity.x;
				spec->main_vector.uy[k] = part_velocity.y;
				spec->main_vector.uz[k] = part_velocity.z;
			}

			// Update the global current with local values
			#pragma acc loop vector collapse(2)
			for(int j = 0; j < (TILE_SIZE + 3); j++)
			{
				for(int i = 0; i < (TILE_SIZE + 3); i++)
				{
					int2 idx;
					idx.x = begin_idx.x + i;
					idx.y = begin_idx.y + j;

					const t_vfld value = J[i + j * (TILE_SIZE + 3)];

					if(idx.x <= spec->nx[0] + 1 && idx.y <= (limits_y[1] - region_offset) + 1)
					{
						#pragma acc atomic
						current->J_buf[idx.x + idx.y * nrow].x += value.x;

						#pragma acc atomic
						current->J_buf[idx.x + idx.y * nrow].y += value.y;

						#pragma acc atomic
						current->J_buf[idx.x + idx.y * nrow].z += value.z;
					}
				}
			}
		}
	}

	// Advance internal iteration number
	spec->iter++;
}

/*********************************************************************************************
 Post Processing
 *********************************************************************************************/

// Shift the particle left and inject particles in the rightmost cells. OpenAcc Task
void spec_move_window_openacc(t_species *restrict spec, const int limits_y[2], const int device)
{
	// Move window
	if (spec->iter * spec->dt > spec->dx[0] * (spec->n_move + 1))
	{
		const int64_t size = spec->main_vector.size;

		// Shift particles left
		#pragma acc parallel loop
		for(int64_t i = 0; i < size; i++)
			if(!spec->main_vector.invalid[i]) spec->main_vector.ix[i]--;

		// Increase moving window counter
		spec->n_move++;

		const int range[][2] = {{spec->nx[0] - 1, spec->nx[0]}, {limits_y[0], limits_y[1]}};
		int64_t np_inj = (range[0][1] - range[0][0]) * (range[1][1] - range[1][0]) * spec->ppc[0] * spec->ppc[1];

		// If needed, add the incoming particles to a temporary vector
		if(!spec->incoming_part[2].enable_vector)
		{
			part_vector_alloc(&spec->incoming_part[2], np_inj, device);
			spec_inject_particles(&spec->incoming_part[2], range, spec->ppc, &spec->density,
					spec->dx, spec->n_move, spec->ufl, spec->uth);
		}else spec->incoming_part[2].size = np_inj; // Reuse the temporary vector (THIS ONLY WORKS IF THE INJECTED PARTICLES HAVE NO MOMENTUM)
	}
}

// Transfer particles between regions (if applicable). OpenAcc Task
void spec_check_boundaries_openacc(t_species *spec, const int limits_y[2], const int device)
{
	const int nx0 = spec->nx[0];
	const int nx1 = spec->nx[1];

#ifdef ENABLE_PREFETCH
	const int queue = nanos6_get_current_acc_queue();
	void *stream = acc_get_cuda_stream(queue);
	spec_prefetch_openacc(spec->outgoing_part[0], device, stream);
	spec_prefetch_openacc(spec->outgoing_part[1], device, stream);
#endif

	// Check if particles are exiting the left boundary (periodic boundary)
	#pragma acc parallel loop gang vector_length(128)
	for(int tile_y = 0; tile_y < spec->n_tiles_y; tile_y++)
	{
		const int tile_idx = tile_y * spec->n_tiles_x;
		const int64_t begin = spec->tile_offset[tile_idx];
		const int64_t end = spec->tile_offset[tile_idx + 1];

		if(spec->moving_window)
		{
			#pragma acc loop vector
			for(int64_t i = begin ; i < end; i++)
				if (spec->main_vector.ix[i] < 0) spec->main_vector.invalid[i] = true;  // Mark the particle as invalid
		}else
		{
			#pragma acc loop vector
			for(int64_t i = begin ; i < end; i++)
				if (spec->main_vector.ix[i] < 0) spec->main_vector.ix[i] += nx0;
		}
	}

	// Check if particles are exiting the right boundary (periodic boundary)
	#pragma acc parallel loop gang vector_length(128)
	for(int tile_y = 0; tile_y < spec->n_tiles_y; tile_y++)
	{
		const int tile_idx = (tile_y + 1) * spec->n_tiles_x - 1;
		const int64_t begin = spec->tile_offset[tile_idx];
		const int64_t end = spec->tile_offset[tile_idx + 1];

		if(spec->moving_window)
		{
			#pragma acc loop vector
			for(int64_t i = begin ; i < end; i++)
				if (spec->main_vector.ix[i] >= nx0) spec->main_vector.invalid[i] = true;  // Mark the particle as invalid
		}else
		{
			#pragma acc loop vector
			for(int64_t i = begin ; i < end; i++)
				if (spec->main_vector.ix[i] >= nx0) spec->main_vector.ix[i] -= nx0;
		}
	}

	// Check if particles are exiting the lower boundary and needs to be transfer to another region
	#pragma acc parallel loop gang
	for (int tile_x = 0; tile_x < spec->n_tiles_x; tile_x++)
	{
		const int64_t begin = spec->tile_offset[tile_x];
		const int64_t end = spec->tile_offset[tile_x + 1];

		#pragma acc loop vector
		for (int64_t i = begin ; i < end; i++)
		{
			bool is_invalid = spec->main_vector.invalid[i];

			if (!is_invalid)
			{
				int iy = spec->main_vector.iy[i];
				int64_t idx;

				// Check if the particle is leaving the box
				if (iy < limits_y[0])
				{
					if (iy < 0) iy += nx1;

					// Reserve a position in the vector
					#pragma acc atomic capture
					idx = spec->outgoing_part[0]->size++;

					spec->outgoing_part[0]->ix[idx] = spec->main_vector.ix[i];
					spec->outgoing_part[0]->iy[idx] = iy;
					spec->outgoing_part[0]->x[idx] = spec->main_vector.x[i];
					spec->outgoing_part[0]->y[idx] = spec->main_vector.y[i];
					spec->outgoing_part[0]->ux[idx] = spec->main_vector.ux[i];
					spec->outgoing_part[0]->uy[idx] = spec->main_vector.uy[i];
					spec->outgoing_part[0]->uz[idx] = spec->main_vector.uz[i];
					spec->outgoing_part[0]->invalid[idx] = false;

					spec->main_vector.invalid[i] = true;  // Mark the particle as invalid
				}
			}
		}
	}

	// Check if particles are exiting the upper boundary and needs to be transfer to another region
	#pragma acc parallel loop gang
	for (int tile_x = 0; tile_x < spec->n_tiles_x; tile_x++)
	{
		const int tile_idx = tile_x + (spec->n_tiles_y - 1) * spec->n_tiles_x;
		const int64_t begin = spec->tile_offset[tile_idx];
		const int64_t end = spec->tile_offset[tile_idx + 1];

		#pragma acc loop vector
		for (int64_t i = begin ; i < end; i++)
		{
			if (!spec->main_vector.invalid[i])
			{
				int iy = spec->main_vector.iy[i];
				int64_t idx;

				// Check if the particle is leaving the box
				if (iy >= limits_y[1])
				{
					if (iy >= nx1) iy -= nx1;

					#pragma acc atomic capture
					idx = spec->outgoing_part[1]->size++;

					spec->outgoing_part[1]->ix[idx] = spec->main_vector.ix[i];
					spec->outgoing_part[1]->iy[idx] = iy;
					spec->outgoing_part[1]->x[idx] = spec->main_vector.x[i];
					spec->outgoing_part[1]->y[idx] = spec->main_vector.y[i];
					spec->outgoing_part[1]->ux[idx] = spec->main_vector.ux[i];
					spec->outgoing_part[1]->uy[idx] = spec->main_vector.uy[i];
					spec->outgoing_part[1]->uz[idx] = spec->main_vector.uz[i];
					spec->outgoing_part[1]->invalid[idx] = false;

					spec->main_vector.invalid[i] = true;  // Mark the particle as invalid
				}
			}
		}
	}
}

/*********************************************************************************************
 Sort
 *********************************************************************************************/

// Exchange particles between tiles (one particle vector parameter at a time to reduce memory usage)
void spec_apply_sort(uint32_t *restrict vector, const int64_t *restrict source_idx,
					 const int64_t *restrict target_idx, uint32_t *restrict temp,
					 const int64_t sort_size, const int queue)
{
	#pragma acc parallel loop deviceptr(temp, source_idx) async(queue)
	for (int64_t i = 0; i < sort_size; i++)
		if (source_idx[i] >= 0)
			temp[i] = vector[source_idx[i]];

	#pragma acc parallel loop deviceptr(temp, source_idx, target_idx) async(queue)
	for (int64_t i = 0; i < sort_size; i++)
		if (source_idx[i] >= 0)
			vector[target_idx[i]] = temp[i];
}

// Calculate an histogram for the number of particles per tile
#pragma oss task label("Sort (GPU, Histogram NP)") device(openacc) \
	inout(tile_offset[0: n_tiles_x * n_tiles_y])
void histogram_np_per_tile(t_part_vector *part_vector, int64_t *restrict tile_offset,
						   t_part_vector incoming_part[3], int64_t *restrict np_per_tile,
						   const int n_tiles_y, const int n_tiles_x, const int offset_region,
						   const int device)
{
	const int n_tiles = n_tiles_x * n_tiles_y;

#ifdef ENABLE_PREFETCH
	const int queue = nanos6_get_current_acc_queue();
	void *stream = acc_get_cuda_stream(queue);
	spec_prefetch_openacc(&incoming_part[0], device, stream);
	spec_prefetch_openacc(&incoming_part[1], device, stream);
#endif

	// Reset the number of particles per tile
	#pragma acc parallel loop deviceptr(np_per_tile)
	for (int i = 0; i < n_tiles; i++)
		np_per_tile[i] = 0;

	// Calculate the histogram (number of particles per tile) for the main vector
	#pragma acc parallel loop gang collapse(2) deviceptr(np_per_tile)
	for(int tile_y = 0; tile_y < n_tiles_y; tile_y++)
	{
		for(int tile_x = 0; tile_x < n_tiles_x; tile_x++)
		{
			const int tile_idx = tile_x + tile_y * n_tiles_x;
			const int64_t begin = tile_offset[tile_idx];
			const int64_t end = tile_offset[tile_idx + 1];

			// Use shared memory to calculate a local histogram
			int np[9];
			#pragma acc cache(np[0: 9])

			#pragma acc loop vector
			for(int i = 0; i < 9; i++)
				np[i] = 0;

			#pragma acc loop vector
			for (int64_t k = begin ; k < end; k++)
			{
				int ix = part_vector->ix[k] / TILE_SIZE;
				int iy = (part_vector->iy[k] - offset_region) / TILE_SIZE;
				bool is_invalid = part_vector->invalid[k];

				int local_ix;
				int local_iy = (iy - tile_y + 1);

				if (tile_x == n_tiles_x - 1 && ix == 0) local_ix = 2;
				else if (tile_x == 0 && ix == n_tiles_x - 1) local_ix = 0;
				else local_ix = (ix - tile_x + 1);

				if (!is_invalid)
				{
					#pragma acc atomic
					np[local_ix + local_iy * 3]++;
				}
			}

			// Add the local values to the a global histogram
			#pragma acc loop vector collapse(2)
			for(int j = 0; j < 3; j++)
			{
				for(int i = 0; i < 3; i++)
				{
					int global_ix = tile_x + i - 1;
					int global_iy = tile_y + j - 1;

					if (global_ix < 0) global_ix += n_tiles_x;
					else if (global_ix >= n_tiles_x) global_ix -= n_tiles_x;

					if(np[i + j * 3] > 0)
					{
						#pragma acc atomic
						np_per_tile[global_ix + global_iy * n_tiles_x] += np[i + j * 3];
					}
				}
			}
		}
	}

	// Add the incoming particles to the histogram
	for(int n = 0; n < 3; n++)
	{
		if(incoming_part[n].enable_vector)
		{
			int64_t size_temp = incoming_part[n].size;

			#pragma acc parallel loop deviceptr(np_per_tile)
			for(int k = 0; k < size_temp; k++)
			{
				int ix = incoming_part[n].ix[k] / TILE_SIZE;
				int iy = (incoming_part[n].iy[k] - offset_region) / TILE_SIZE;
				int target_tile = ix + iy * n_tiles_x;

				#pragma acc atomic
				np_per_tile[target_tile]++;
			}
		}
	}

	// Copy the histogram to calculate the new tile offset
	#pragma acc parallel loop deviceptr(np_per_tile)
	for (int i = 0; i <= n_tiles; i++)
		if(i < n_tiles) tile_offset[i] = np_per_tile[i];
		else tile_offset[i] = 0;
}

// Calculate an histogram for the particles moving between tiles
#pragma oss task label("Sort (GPU, Histogram Leaving Part)") device(openacc)  \
	in(tile_offset[0: n_tiles]) out(np_leaving[0: n_tiles])
void histogram_moving_particles(t_part_vector *part_vector, int64_t *restrict tile_offset,
								int64_t *restrict np_leaving, const int n_tiles, const int n_tiles_x,
								const int offset_region, const int64_t old_size)
{
	#pragma acc parallel loop gang
	for(int tile_idx = 0; tile_idx < n_tiles; tile_idx++)
	{
		const int64_t begin = tile_offset[tile_idx];
		const int64_t end = tile_offset[tile_idx + 1];
		int leaving_count = 0;

		#pragma acc loop vector reduction(+ : leaving_count)
		for (int64_t k = begin ; k < end; k++)
		{
			if(k >= old_size) part_vector->invalid[k] = true;

			int ix = part_vector->ix[k] / TILE_SIZE;
			int iy = (part_vector->iy[k] - offset_region) / TILE_SIZE;
			bool is_invalid = part_vector->invalid[k];
			int target_tile = ix + iy * n_tiles_x;

			if (is_invalid || target_tile != tile_idx) leaving_count++;
		}

		np_leaving[tile_idx] = leaving_count;
	}
}

// Identify the particles in the wrong tile and then generate a sorted list for them
// source idx - particle's current position / target idx - particle's new position
void calculate_sorted_idx(t_part_vector *part_vector, int64_t *restrict tile_offset,
						  int64_t *restrict source_idx, int64_t *restrict target_idx,
						  int64_t *restrict sort_counter, int64_t *restrict mv_part_offset,
						  const int n_tiles_y, const int n_tiles_x, const int64_t old_size,
						  const int offset_region, const int64_t sorting_size, const int queue)
{
	const int n_tiles = n_tiles_x * n_tiles_y;
	const int64_t size = part_vector->size;

	#pragma acc parallel loop deviceptr(source_idx) async(queue)
	for(int64_t i = 0; i < sorting_size; i++)
		source_idx[i] = -1;

	#pragma acc parallel loop deviceptr(sort_counter) async(queue)
	for (int i = 0; i < n_tiles; i++)
		sort_counter[i] = mv_part_offset[i];

	// Determine which particles are in the wrong tile
	#pragma acc parallel loop gang collapse(2) deviceptr(target_idx, sort_counter) async(queue)
	for(int tile_y = 0; tile_y < n_tiles_y; tile_y++)
	{
		for(int tile_x = 0; tile_x < n_tiles_x; tile_x++)
		{
			const int tile_idx = tile_x + tile_y * n_tiles_x;
			const int64_t begin = tile_offset[tile_idx];
			const int64_t end = tile_offset[tile_idx + 1];
			int64_t offset = mv_part_offset[tile_idx];

			int64_t right_counter = 0; // Count the particles moving to the right tile

			#pragma acc loop vector reduction(+ : right_counter)
			for (int64_t k = begin ; k < end; k++)
			{
				int64_t idx;

				int ix = part_vector->ix[k] / TILE_SIZE;
				int iy = (part_vector->iy[k] - offset_region) / TILE_SIZE;
				bool is_invalid = part_vector->invalid[k];
				int target_tile = ix + iy * n_tiles_x;

				if (is_invalid || target_tile != tile_idx)
				{
					#pragma acc atomic capture
					idx = offset++;

					target_idx[idx] = k;
				}

				if(!is_invalid && target_tile == tile_idx + 1)
					right_counter++;
			}

			if(tile_x < n_tiles_x - 1)
			{
				#pragma acc atomic
				sort_counter[tile_idx + 1] += right_counter; // Create a space for incoming particles from the current tile to its right tile
			}
		}
	}

	// Generate a sorted list for the particles in the wrong tile
	#pragma acc parallel loop gang collapse(2) deviceptr(source_idx, target_idx, sort_counter) async(queue)
	for(int tile_y = 0; tile_y < n_tiles_y; tile_y++)
	{
		for(int tile_x = 0; tile_x < n_tiles_x; tile_x++)
		{
			const int tile_idx = tile_x + tile_y * n_tiles_x;
			const int64_t begin = mv_part_offset[tile_idx];
			const int64_t end = mv_part_offset[tile_idx + 1];

			int64_t left_counter = begin - 1; // Local counter for the particle going to the left tile
			int64_t right_counter = end; // Local counter for the particle going to the right tile

			#pragma acc loop vector
			for (int64_t k = begin ; k < end; k++)
			{
				int64_t idx;
				int64_t source = target_idx[k];
				int ix = part_vector->ix[source] / TILE_SIZE;
				int iy = (part_vector->iy[source] - offset_region) / TILE_SIZE;
				bool is_invalid = part_vector->invalid[source];

				int target_tile = ix + iy * n_tiles_x;

				if (!is_invalid)
				{
					if (tile_x > 0 && target_tile == tile_idx - 1)
					{
						#pragma acc atomic capture
						idx = left_counter--;
					}
					else if (tile_x < n_tiles_x - 1 && target_tile == tile_idx + 1)
					{
						#pragma acc atomic capture
						idx = right_counter++;
					}else
					{
						#pragma acc atomic capture
						idx = sort_counter[target_tile]++;
					}

					source_idx[idx] = source;
				}
			}
		}
	}

	// If the vector has shrink, add the valid particles outside the vector new size
	if(size < old_size)
	{
		#pragma acc parallel loop deviceptr(source_idx, sort_counter) async(queue)
		for(int64_t k = size; k < old_size; k++)
		{
			int64_t idx;
			int ix = part_vector->ix[k] / TILE_SIZE;
			int iy = (part_vector->iy[k] - offset_region) / TILE_SIZE;
			bool is_invalid = part_vector->invalid[k];

			int target_tile = ix + iy * n_tiles_x;

			if (!is_invalid)
			{
				#pragma acc atomic capture
				idx = sort_counter[target_tile]++;

				source_idx[idx] = k;
			}
		}
	}
}

// Merge the temporary vector for the incoming particle into the main vector
void merge_particles_buffers(t_part_vector *part_vector, t_part_vector *incoming_part,
							 int64_t *restrict counter, int64_t *restrict target_idx, const int n_tiles_x,
							 const int offset_region, const int queue)
{
	for(int n = 0; n < 3; n++)
	{
		if(incoming_part[n].enable_vector)
		{
			int64_t size_temp = incoming_part[n].size;

			#pragma acc parallel loop firstprivate(size_temp) deviceptr(target_idx, counter) async(queue)
			for(int64_t k = 0; k < size_temp; k++)
			{
				int64_t idx;
				int ix = incoming_part[n].ix[k] / TILE_SIZE;
				int iy = (incoming_part[n].iy[k] - offset_region) / TILE_SIZE;
				int target_tile = ix + iy * n_tiles_x;

				#pragma acc atomic capture
				idx = counter[target_tile]++;

				int64_t target = target_idx[idx];

				part_vector->ix[target] = incoming_part[n].ix[k];
				part_vector->iy[target] = incoming_part[n].iy[k];
				part_vector->x[target] = incoming_part[n].x[k];
				part_vector->y[target] = incoming_part[n].y[k];
				part_vector->ux[target] = incoming_part[n].ux[k];
				part_vector->uy[target] = incoming_part[n].uy[k];
				part_vector->uz[target] = incoming_part[n].uz[k];
				part_vector->invalid[target] = false;
			}

			incoming_part[n].size = 0;
		}
	}
}

#pragma oss task label("Sort (GPU, Sort Particles)") device(openacc) \
	inout(tile_offset[0: n_tiles_x * n_tiles_y]) in(mv_part_offset[0: n_tiles_x * n_tiles_y])
void spec_sort_particles(t_part_vector *part_vector, t_part_vector incoming_part[3],
						 int64_t *tile_offset, int64_t *mv_part_offset, int64_t *source_idx, int64_t *target_idx,
						 int64_t *counter, uint32_t *temp, const int n_tiles_x, const int n_tiles_y,
						 const int offset_region, const int64_t old_size)
{
	const int queue = nanos6_get_current_acc_queue();

	int n_tiles = n_tiles_x * n_tiles_y;
	int64_t sorting_size = mv_part_offset[n_tiles];
	part_vector->size = tile_offset[n_tiles];

	if(sorting_size >= MAX_LEAVING_PART * part_vector->size_max)
	{
		fprintf(stderr, "Out-of-bounds: Buffer size (%d) is too small (Minimum: %d)."
				"Increase MAX_LEAVING_PART to solve this issue.\n",
				MAX_LEAVING_PART * part_vector->size_max, sorting_size);
		exit(1);
	}

	calculate_sorted_idx(part_vector, tile_offset, source_idx, target_idx, counter, mv_part_offset,
						 n_tiles_y, n_tiles_x, old_size, offset_region, sorting_size, queue);

	spec_apply_sort((uint32_t*) part_vector->ix, source_idx, target_idx, temp, sorting_size, queue);
	spec_apply_sort((uint32_t*) part_vector->iy, source_idx, target_idx, temp, sorting_size, queue);
	spec_apply_sort((uint32_t*) part_vector->x, source_idx, target_idx, temp, sorting_size, queue);
	spec_apply_sort((uint32_t*) part_vector->y, source_idx, target_idx, temp, sorting_size, queue);
	spec_apply_sort((uint32_t*) part_vector->ux, source_idx, target_idx, temp, sorting_size, queue);
	spec_apply_sort((uint32_t*) part_vector->uy, source_idx, target_idx, temp, sorting_size, queue);
	spec_apply_sort((uint32_t*) part_vector->uz, source_idx, target_idx, temp, sorting_size, queue);

	#pragma acc parallel loop deviceptr(source_idx, target_idx)
	for(int64_t i = 0; i < sorting_size; i++)
		if(source_idx[i] >= 0)
			part_vector->invalid[target_idx[i]] = false;

	merge_particles_buffers(part_vector, incoming_part, counter, target_idx, n_tiles_x,
							offset_region, queue);
}

void spec_sort_openacc(t_species *spec, const int limits_y[2], const int device)
{
	const int64_t old_size = spec->main_vector.size;
	const int n_tiles = spec->n_tiles_x * spec->n_tiles_y;
	spec->mv_part_offset[n_tiles] = 0;

	acc_set_device_num(device, DEVICE_TYPE);

	const int64_t max_leaving_np = MAX_LEAVING_PART * spec->main_vector.size_max;
	int64_t *restrict source_idx = acc_malloc(max_leaving_np * sizeof(int64_t));
	int64_t *restrict target_idx = acc_malloc(max_leaving_np * sizeof(int64_t));
	int64_t *restrict counter = acc_malloc(n_tiles * sizeof(int64_t));
	int64_t *restrict np_per_tile = acc_malloc(n_tiles * sizeof(int64_t));
	uint32_t *restrict temp = acc_malloc(max_leaving_np * sizeof(uint32_t));

	int64_t np_inj = 0;
	for (int i = 0; i < 3; i++)
		if (spec->incoming_part[i].enable_vector)
			np_inj += spec->incoming_part[i].size;

	// Check if buffer is large enough and if not reallocate
	if (spec->main_vector.size + np_inj > spec->main_vector.size_max)
	{
		const int new_size = ((spec->main_vector.size_max + np_inj) / 1024 + 1) * 1024;
		part_vector_realloc(&spec->main_vector, new_size, device);
	}

	// Calculate the new offset (in the particle vector) for the tiles
	histogram_np_per_tile(&spec->main_vector, spec->tile_offset, spec->incoming_part, np_per_tile,
						  spec->n_tiles_y, spec->n_tiles_x, limits_y[0], device);

	#pragma oss task inout(spec->tile_offset[0: n_tiles])
	prefix_sum_serial(spec->tile_offset, n_tiles + 1);

	// Calculate the offset for the moving particles between tiles
	histogram_moving_particles(&spec->main_vector, spec->tile_offset, spec->mv_part_offset, n_tiles,
							   spec->n_tiles_x, limits_y[0], old_size);

	#pragma oss task inout(spec->mv_part_offset[0: n_tiles])
	prefix_sum_serial(spec->mv_part_offset, n_tiles + 1);

	spec_sort_particles(&spec->main_vector, spec->incoming_part, spec->tile_offset,
						spec->mv_part_offset, source_idx, target_idx, counter, temp,
						spec->n_tiles_x, spec->n_tiles_y, limits_y[0], old_size);

	#pragma oss taskwait on(spec->tile_offset[0 : n_tiles])
	acc_set_device_num(device, DEVICE_TYPE);

	acc_free(np_per_tile);
	acc_free(temp);
	acc_free(counter);
	acc_free(target_idx);
	acc_free(source_idx);
}

/*********************************************************************************************
 Charge Deposition
 *********************************************************************************************/
// Deposit the particle charge over the simulation grid
void spec_deposit_charge(const t_species *spec, t_part_data *charge)
{
	// Charge array is expected to have 1 guard cell at the upper boundary
	int nrow = spec->nx[0] + 1;
	t_part_data q = spec->q;

	for (int64_t i = 0; i < spec->main_vector.size; i++)
	{
		if (spec->main_vector.invalid[i]) continue;

		int idx = spec->main_vector.ix[i] + nrow * spec->main_vector.iy[i];
		t_fld w1, w2;

		w1 = spec->main_vector.x[i];
		w2 = spec->main_vector.y[i];

		charge[idx] += (1.0f - w1) * (1.0f - w2) * q;
		charge[idx + 1] += (w1) * (1.0f - w2) * q;
		charge[idx + nrow] += (1.0f - w1) * (w2) * q;
		charge[idx + 1 + nrow] += (w1) * (w2) * q;
	}

}

/*********************************************************************************************
 Diagnostics
 *********************************************************************************************/

// Save the deposit particle charge in ZDF file
void spec_rep_charge(t_part_data *restrict charge, const int true_nx[2], const t_fld box[2],
		const int iter_num, const float dt, const bool moving_window, const char path[128])
{
	size_t buf_size = true_nx[0] * true_nx[1] * sizeof(t_part_data);
	t_part_data *restrict buf = malloc(buf_size);

	// Correct boundary values
	// x
	if (!moving_window) for (int j = 0; j < true_nx[1] + 1; j++)
		charge[0 + j * (true_nx[0] + 1)] += charge[true_nx[0] + j * (true_nx[0] + 1)];

	// y - Periodic boundaries
	for (int i = 0; i < true_nx[0] + 1; i++)
		charge[i] += charge[i + true_nx[1] * (true_nx[0] + 1)];

	t_part_data *restrict b = buf;
	t_part_data *restrict c = charge;

	for (int j = 0; j < true_nx[1]; j++)
	{
		for (int i = 0; i < true_nx[0]; i++)
			b[i] = c[i];

		b += true_nx[0];
		c += true_nx[0] + 1;
	}

	t_zdf_grid_axis axis[2];
	axis[0] = (t_zdf_grid_axis) {.min = 0.0, .max = box[0], .label = "x_1",
									.units = "c/\\omega_p"};

	axis[1] = (t_zdf_grid_axis) {.min = 0.0, .max = box[1], .label = "x_2",
									.units = "c/\\omega_p"};

	t_zdf_grid_info info = {.ndims = 2, .label = "charge", .units = "n_e", .axis = axis};

	info.nx[0] = true_nx[0];
	info.nx[1] = true_nx[1];

	t_zdf_iteration iter = {.n = iter_num, .t = iter_num * dt, .time_units = "1/\\omega_p"};

	zdf_save_grid(buf, &info, &iter, path);

	free(buf);

}

void spec_pha_axis(const t_species *spec, int i0, int np, int quant, float *axis)
{
	int i;

	switch (quant)
	{
		case X1:
			for (i = 0; i < np; i++)
				axis[i] = (spec->main_vector.x[i0 + i] + spec->main_vector.ix[i0 + i])
						* spec->dx[0];
			break;
		case X2:
			for (i = 0; i < np; i++)
				axis[i] = (spec->main_vector.y[i0 + i] + spec->main_vector.iy[i0 + i])
						* spec->dx[1];
			break;
		case U1:
			for (i = 0; i < np; i++)
				axis[i] = spec->main_vector.ux[i0 + i];
			break;
		case U2:
			for (i = 0; i < np; i++)
				axis[i] = spec->main_vector.uy[i0 + i];
			break;
		case U3:
			for (i = 0; i < np; i++)
				axis[i] = spec->main_vector.uz[i0 + i];
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

// Deposit the particles over the phase space
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

	for (int64_t i = 0; i < spec->main_vector.size; i += BUF_SIZE)
	{
		int64_t np = (i + BUF_SIZE > spec->main_vector.size) ? spec->main_vector.size - i : BUF_SIZE;

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

// Save the phase space in a ZDF file
void spec_rep_pha(const t_part_data *buffer, const int rep_type, const int pha_nx[],
		const float pha_range[][2], const int iter_num, const float dt, const char path[128])
{
	char const *const pha_ax_name[] = {"x1", "x2", "x3", "u1", "u2", "u3"};
	char pha_name[64];

	// save the data in hdf5 format
	int quant1 = rep_type & 0x000F;
	int quant2 = (rep_type & 0x00F0) >> 4;

	const char *pha_ax1_units = spec_pha_axis_units(quant1);
	const char *pha_ax2_units = spec_pha_axis_units(quant2);

	sprintf(pha_name, "%s%s", pha_ax_name[quant1 - 1], pha_ax_name[quant2 - 1]);

	t_zdf_grid_axis axis[2];
	axis[0] = (t_zdf_grid_axis) {.min = pha_range[0][0], .max = pha_range[0][1],
									.label = (char*) pha_ax_name[quant1 - 1],
									.units = (char*) pha_ax1_units};

	axis[1] = (t_zdf_grid_axis) {.min = pha_range[1][0], .max = pha_range[1][1],
									.label = (char*) pha_ax_name[quant2 - 1],
									.units = (char*) pha_ax2_units};

	t_zdf_grid_info info = {.ndims = 2, .label = pha_name, .units = "a.u.", .axis = axis};

	info.nx[0] = pha_nx[0];
	info.nx[1] = pha_nx[1];

	t_zdf_iteration iter = {.n = iter_num, .t = iter_num * dt, .time_units = "1/\\omega_p"};

	zdf_save_grid(buffer, &info, &iter, path);
}

// Calculate time centre energy
void spec_calculate_energy(t_species *spec)
{
	spec->energy = 0;

	for (int64_t i = 0; i < spec->main_vector.size; i++)
	{
		t_part_data usq = spec->main_vector.ux[i] * spec->main_vector.ux[i]
				+ spec->main_vector.uy[i] * spec->main_vector.uy[i]
				+ spec->main_vector.uz[i] * spec->main_vector.uz[i];
		t_part_data gamma = sqrtf(1 + usq);
		spec->energy += usq / (gamma + 1.0);
	}
}
