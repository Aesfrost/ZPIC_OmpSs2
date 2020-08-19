/*********************************************************************************************
 ZPIC
 particles.c

 Created by Ricardo Fonseca on 11/8/10.
 Modified by Nicolas Guidotti on 11/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#include "particles.h"

#include "random.h"
#include "emf.h"
#include "current.h"
#include "utilities.h"
#include "zdf.h"
#include "timer.h"

static double _spec_time = 0.0;
static double _spec_npush = 0.0;

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
 * Vector Utilities
 *********************************************************************************************/

void part_vector_alloc(t_particle_vector *vector, const int size_max)
{
	 vector->ix = alloc_align_buffer(DEFAULT_ALIGNMENT, size_max * sizeof(int));
	 vector->iy = alloc_align_buffer(DEFAULT_ALIGNMENT, size_max * sizeof(int));
	 vector->x = alloc_align_buffer(DEFAULT_ALIGNMENT, size_max * sizeof(t_part_data));
	 vector->y = alloc_align_buffer(DEFAULT_ALIGNMENT, size_max * sizeof(t_part_data));
	 vector->ux = alloc_align_buffer(DEFAULT_ALIGNMENT, size_max * sizeof(t_part_data));
	 vector->uy = alloc_align_buffer(DEFAULT_ALIGNMENT, size_max * sizeof(t_part_data));
	 vector->uz = alloc_align_buffer(DEFAULT_ALIGNMENT, size_max * sizeof(t_part_data));
	 vector->invalid = alloc_align_buffer(DEFAULT_ALIGNMENT, size_max * sizeof(bool));

	 vector->tile_offset = NULL;
	 vector->size_max = size_max;
	 vector->size = 0;
}

void part_vector_free(t_particle_vector *vector)
{
	 free_align_buffer(vector->ix);
	 free_align_buffer(vector->iy);
	 free_align_buffer(vector->x);
	 free_align_buffer(vector->y);
	 free_align_buffer(vector->ux);
	 free_align_buffer(vector->uy);
	 free_align_buffer(vector->uz);
	 free_align_buffer(vector->invalid);

	 if (vector->tile_offset) free(vector->tile_offset);
}

void part_vector_realloc(t_particle_vector *vector, const int new_size)
{
	 vector->size_max = new_size;

	 realloc_align_buffer((void**) &vector->ix, vector->size, vector->size_max, sizeof(int),
						  DEFAULT_ALIGNMENT);
	 realloc_align_buffer((void**) &vector->iy, vector->size, vector->size_max, sizeof(int),
						  DEFAULT_ALIGNMENT);
	 realloc_align_buffer((void**) &vector->x, vector->size, vector->size_max, sizeof(t_part_data),
						  DEFAULT_ALIGNMENT);
	 realloc_align_buffer((void**) &vector->y, vector->size, vector->size_max, sizeof(t_part_data),
						  DEFAULT_ALIGNMENT);
	 realloc_align_buffer((void**) &vector->ux, vector->size, vector->size_max, sizeof(t_part_data),
						  DEFAULT_ALIGNMENT);
	 realloc_align_buffer((void**) &vector->uy, vector->size, vector->size_max, sizeof(t_part_data),
						  DEFAULT_ALIGNMENT);
	 realloc_align_buffer((void**) &vector->uz, vector->size, vector->size_max, sizeof(t_part_data),
						  DEFAULT_ALIGNMENT);
	 realloc_align_buffer((void**) &vector->invalid, vector->size, vector->size_max, sizeof(bool),
						  DEFAULT_ALIGNMENT);
}

void part_vector_assign_valid_part(const t_particle_vector *source, const int source_idx,
									t_particle_vector *target, const int target_idx)
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

void part_vector_memcpy(const t_particle_vector *source, t_particle_vector *target, const int begin,
						 const int size)
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

/*********************************************************************************************
 Vector Handling
 *********************************************************************************************/
/**
 * Add to the content of the temporary buffers into the particles vector
 * @param spec  Particle species
 */
void spec_update_main_vector(t_species *spec)
{
	int i = 0, j, k;
	int size = spec->main_vector.size;

	const int np_inj = spec->incoming_part[0].size + spec->incoming_part[1].size;

	// Check if buffer is large enough and if not reallocate
	if (spec->main_vector.size + np_inj > spec->main_vector.size_max)
		part_vector_realloc(&spec->main_vector, ((spec->main_vector.size_max + np_inj) / 1024 + 1) * 1024);

	//Loop through all the 2 temp buffers
	for (k = 0; k < 2; k++)
	{
		int size_temp = spec->incoming_part[k].size;

		//Loop through all elements on the buffer, copying to the main_vector particle buffer (if applicable)
		for (j = 0; j < size_temp; j++)
		{
			while (i < size && !spec->main_vector.invalid[i])
				i++;   //Checks if a particle can be safely deleted
			if (i < size)
			{
				part_vector_assign_valid_part(&spec->incoming_part[k], j, &spec->main_vector, i);
				i++;
			} else
			{
				part_vector_assign_valid_part(&spec->incoming_part[k], j, &spec->main_vector,
												spec->main_vector.size);
				spec->main_vector.size++;
			}
		}

	}

	if (i < size)
	{
		while (i < spec->main_vector.size)
		{
			if (spec->main_vector.invalid[i])
			{
				spec->main_vector.size--;
				part_vector_assign_valid_part(&spec->main_vector, spec->main_vector.size, &spec->main_vector, i);
			}
			else i++;
		}
	}

	//Clean the temp buffer
	for (k = 0; k < 2; k++)
		spec->incoming_part[k].size = 0;
}

/*********************************************************************************************
 Initialization
 *********************************************************************************************/

/**
 * Sets the momentum of the range of particles supplieds using a thermal distribution
 * @param spec  Particle species
 * @param start Index of the first particle to set the momentum
 * @param end   Index of the last particle to set the momentum
 */
void spec_set_u(t_species *spec, const int start, const int end)
{
	for (int i = start; i <= end; i++)
	{
		spec->main_vector.ux[i] = spec->ufl[0] + spec->uth[0] * rand_norm();
		spec->main_vector.uy[i] = spec->ufl[1] + spec->uth[1] * rand_norm();
		spec->main_vector.uz[i] = spec->ufl[2] + spec->uth[2] * rand_norm();
	}
}

void spec_set_x(t_species *spec, const int range[][2])
{
	int i, j, k, ip;

	float *poscell;
	float start, end;

	// Calculate particle positions inside the cell
	const int npc = spec->ppc[0] * spec->ppc[1];
	t_part_data const dpcx = 1.0f / spec->ppc[0];
	t_part_data const dpcy = 1.0f / spec->ppc[1];

	poscell = malloc(2 * npc * sizeof(t_part_data));
	ip = 0;
	for (j = 0; j < spec->ppc[1]; j++)
	{
		for (i = 0; i < spec->ppc[0]; i++)
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

			for (j = range[1][0]; j < range[1][1]; j++)
			{
				for (i = range[0][0]; i < range[0][1]; i++)
				{
					for (k = 0; k < npc; k++)
					{
						if (i + poscell[2 * k] > start)
						{
							spec->main_vector.ix[ip] = i;
							spec->main_vector.iy[ip] = j;
							spec->main_vector.x[ip] = poscell[2 * k];
							spec->main_vector.y[ip] = poscell[2 * k + 1];
							spec->main_vector.invalid[ip] = false;
							ip++;
						}
					}
				}
			}
			break;

		case SLAB:    // Slab like density profile

			// Get edge position normalized to cell size;
			start = spec->density.start / spec->dx[0] - spec->n_move;
			end = spec->density.end / spec->dx[0] - spec->n_move;

			for (j = range[1][0]; j < range[1][1]; j++)
			{
				for (i = range[0][0]; i < range[0][1]; i++)
				{
					for (k = 0; k < npc; k++)
					{
						if (i + poscell[2 * k] > start && i + poscell[2 * k] < end)
						{
							spec->main_vector.ix[ip] = i;
							spec->main_vector.iy[ip] = j;
							spec->main_vector.x[ip] = poscell[2 * k];
							spec->main_vector.y[ip] = poscell[2 * k + 1];
							spec->main_vector.invalid[ip] = false;
							ip++;
						}
					}
				}
			}
			break;

		default:    // Uniform density
			for (j = range[1][0]; j < range[1][1]; j++)
			{
				for (i = range[0][0]; i < range[0][1]; i++)
				{
					for (k = 0; k < npc; k++)
					{
						spec->main_vector.ix[ip] = i;
						spec->main_vector.iy[ip] = j;
						spec->main_vector.x[ip] = poscell[2 * k];
						spec->main_vector.y[ip] = poscell[2 * k + 1];
						spec->main_vector.invalid[ip] = false;
						ip++;
					}
				}
			}
	}

	spec->main_vector.size = ip;
	free(poscell);
}

// Inject the particles in the simulation
void spec_inject_particles(t_species *spec, const int range[][2])
{
	int start = spec->main_vector.size;

	// Get maximum number of particles to inject
	int np_inj = (range[0][1] - range[0][0]) * (range[1][1] - range[1][0]) * spec->ppc[0] * spec->ppc[1];

	// Check if buffer is large enough and if not reallocate
	if (spec->main_vector.size + np_inj > spec->main_vector.size_max)
		part_vector_realloc(&spec->main_vector, ((spec->main_vector.size_max + np_inj) / 1024 + 1) * 1024);

	// Set particle positions
	spec_set_x(spec, range);

	// Set momentum of injected particles
	spec_set_u(spec, start, spec->main_vector.size - 1);
}

// Constructor
void spec_new(t_species *spec, char name[], const t_part_data m_q, const int ppc[],
		const t_part_data *ufl, const t_part_data *uth, const int nx[], t_part_data box[],
		const float dt, t_density *density, const int region_size)
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

	// Initialize particle buffer
	part_vector_alloc(&spec->main_vector, (npc * region_size * nx[0] / 1024 + 1) * 1024);

	// Initialize temp buffer
	for (i = 0; i < 2; i++)
		part_vector_alloc(&spec->incoming_part[i], (npc * nx[0] / 1024 + 1) * 1024);

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
}

void spec_delete(t_species *spec)
{
	part_vector_free(&spec->main_vector);

	for(int n = 0; n < 2; n++)
		part_vector_free(&spec->incoming_part[n]);
}

/*********************************************************************************************
 Current deposition
 *********************************************************************************************/
// Current deposition (Esirkepov method)
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

// Current deposition (adapted Villasenor-Bunemann method). CPU Task
void dep_current_zamb(int ix, int iy, int di, int dj, float x0, float y0, float dx, float dy,
		float qnx, float qny, float qvz, t_current *current)
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
	t_vfld *restrict const J = current->J;

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
 Sort
 *********************************************************************************************/

/*********************************************************************************************
 Particle advance
 *********************************************************************************************/
void interpolate_fld(const t_vfld *restrict const E, const t_vfld *restrict const B,
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

// Particle advance~(CPU)
void spec_advance(t_species *spec, t_emf *emf, t_current *current, int limits_y[2])
{
	int i;
	t_part_data qnx, qny, qvz;

	const t_part_data tem = 0.5 * spec->dt / spec->m_q;
	const t_part_data dt_dx = spec->dt / spec->dx[0];
	const t_part_data dt_dy = spec->dt / spec->dx[1];

	// Auxiliary values for current deposition
	qnx = spec->q * spec->dx[0] / spec->dt;
	qny = spec->q * spec->dx[1] / spec->dt;

	// Advance internal iteration number
	spec->iter += 1;

	// Advance particles
	for (i = 0; i < spec->main_vector.size; i++)
	{
		if(spec->main_vector.invalid[i]) continue;

		t_vfld Ep, Bp;
		t_part_data utx, uty, utz;
		t_part_data ux, uy, uz, rg;
		t_part_data gtem, otsq;

		t_part_data x1, y1;

		int di, dj;
		float dx, dy;

		// Load particle momenta
		ux = spec->main_vector.ux[i];
		uy = spec->main_vector.uy[i];
		uz = spec->main_vector.uz[i];

		// Interpolate fields
		interpolate_fld(emf->E, emf->B, emf->nrow, spec->main_vector.ix[i],
								spec->main_vector.iy[i] - limits_y[0], spec->main_vector.x[i],
								spec->main_vector.y[i], &Ep, &Bp);
		// Advance u using Boris scheme
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
		spec->main_vector.ux[i] = ux;
		spec->main_vector.uy[i] = uy;
		spec->main_vector.uz[i] = uz;

		// push particle
		rg = 1.0f / sqrtf(1.0f + ux * ux + uy * uy + uz * uz);

		dx = dt_dx * rg * ux;
		dy = dt_dy * rg * uy;

		x1 = spec->main_vector.x[i] + dx;
		y1 = spec->main_vector.y[i] + dy;

		di = LTRIM(x1);
		dj = LTRIM(y1);

		x1 -= di;
		y1 -= dj;

		qvz = spec->q * uz * rg;

		dep_current_zamb(spec->main_vector.ix[i], spec->main_vector.iy[i] - limits_y[0],
				di, dj, spec->main_vector.x[i], spec->main_vector.y[i], dx, dy, qnx, qny,
				qvz, current);

		// Store results
		spec->main_vector.x[i] = x1;
		spec->main_vector.y[i] = y1;
		spec->main_vector.ix[i] += di;
		spec->main_vector.iy[i] += dj;
	}
}

// Particle post processing (Transfer particles between regions and move the simulation
// window, if applicable). CPU Task
void spec_post_processing(t_species *spec, t_species *upper_spec, t_species *lower_spec,
		int limits_y[2])
{
	const int nx0 = spec->nx[0];
	const int nx1 = spec->nx[1];

	for(int i = 0; i < spec->main_vector.size; i++)
	{
		//Check if the particle is in the correct region
		int iy = spec->main_vector.iy[i];

		// First shift particle left (if applicable), then check for particles leaving the box
		if (spec->moving_window)
		{
			if ((spec->iter * spec->dt) > (spec->dx[0] * (spec->n_move + 1)))
				spec->main_vector.ix[i]--;

			if ((spec->main_vector.ix[i] < 0) || (spec->main_vector.ix[i] >= nx0))
			{
				spec->main_vector.invalid[i] = true;
				continue;
			}
		} else
		{
			//Periodic boundaries for both axis
			if (spec->main_vector.ix[i] < 0) spec->main_vector.ix[i] += nx0;
			else if (spec->main_vector.ix[i] >= nx0) spec->main_vector.ix[i] -= nx0;
		}

		if (spec->main_vector.iy[i] < 0) spec->main_vector.iy[i] += nx1;
		else if (spec->main_vector.iy[i] >= nx1) spec->main_vector.iy[i] -= nx1;

		//Verify if the particle is still in the correct region. If not send the particle to the correct one
		if (iy < limits_y[0])
		{
			part_vector_assign_valid_part(&spec->main_vector, i, &lower_spec->incoming_part[1], lower_spec->incoming_part[1].size);
			lower_spec->incoming_part[1].size++;
			spec->main_vector.invalid[i] = true; // Mark the particle as invalid

		} else if (iy >= limits_y[1])
		{
			part_vector_assign_valid_part(&spec->main_vector, i, &upper_spec->incoming_part[0], upper_spec->incoming_part[0].size);
			upper_spec->incoming_part[0].size++;
			spec->main_vector.invalid[i] = true; // Mark the particle as invalid
		}
	}

	if (spec->moving_window && (spec->iter * spec->dt) > (spec->dx[0] * (spec->n_move + 1)))
	{
		// Increase moving window counter
		spec->n_move++;

		// Inject particles in the right edge of the simulation box
		const int range[][2] = {{spec->nx[0] - 1, spec->nx[0]}, {limits_y[0], limits_y[1]}};
		spec_inject_particles(spec, range);
	}

	//if(spec->iter % 20 == 0) spec_sort(spec, 4);
}

/*********************************************************************************************
 Charge Deposition
 *********************************************************************************************/
// Deposit the particle charge over the simulation grid
void spec_deposit_charge(const t_species *spec, t_part_data *charge)
{
	int i;

	// Charge array is expected to have 1 guard cell at the upper boundary
	int nrow = spec->nx[0] + 1;
	t_part_data q = spec->q;

	for (i = 0; i < spec->main_vector.size; i++)
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

	for (int i = 0; i < spec->main_vector.size; i += BUF_SIZE)
	{
		int np = (i + BUF_SIZE > spec->main_vector.size) ? spec->main_vector.size - i : BUF_SIZE;

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

	for (int i = 0; i < spec->main_vector.size; i++)
	{
		t_part_data usq = spec->main_vector.ux[i] * spec->main_vector.ux[i]
				+ spec->main_vector.uy[i] * spec->main_vector.uy[i]
				+ spec->main_vector.uz[i] * spec->main_vector.uz[i];
		t_part_data gamma = sqrtf(1 + usq);
		spec->energy += usq / (gamma + 1.0);
	}
}