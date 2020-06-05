#include "region.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "timer.h"

static int _n_regions = 0;
static int _effective_gpu_regions = 0;

int get_n_regions()
{
	return _n_regions;
}

int get_gpu_regions_effective()
{
	return _effective_gpu_regions;
}

/*********************************************************************************************
 Initialisation
 *********************************************************************************************/
void region_new(t_region *region, int n_regions, int nx[2], int id, int n_spec, t_species *spec,
		float box[], float dt, float gpu_percentage, int n_gpu_regions, t_region *prev_region)
{
	region->id = id;
	region->prev = prev_region;

	region->iter = 0;

	int gpu_step = floor((float) gpu_percentage * n_regions / n_gpu_regions); // The number of regions to merge

	if(region->id < n_gpu_regions * gpu_step)
	{
		region->limits_y[0] = floor((float) id * nx[1] / n_regions);

		id += gpu_step - 1;
		region->limits_y[1] = floor((float) (id + 1) * nx[1] / n_regions);
		if(region->limits_y[1] > nx[1]) region->limits_y[1] = nx[1];

		region->enable_gpu = true;

	}else
	{
		region->limits_y[0] = floor((float) id * nx[1] / n_regions);
		region->limits_y[1] = floor((float) (id + 1) * nx[1] / n_regions);
		region->enable_gpu = false;
	}

	region->nx[0] = nx[0];
	region->nx[1] = region->limits_y[1] - region->limits_y[0];

	region->iter_time = 0.0;

	// Initialise particles in the region
	t_particle_vector *restrict particles;

	region->n_species = n_spec;
	region->species = (t_species*) malloc(n_spec * sizeof(t_species));
	assert(region->species);

	#pragma acc set device_num(0)
	int start, end;
	for (int n = 0; n < n_spec; ++n)
	{
		spec_new(&region->species[n], spec[n].name, spec[n].m_q, spec[n].ppc, spec[n].ufl,
				spec[n].uth, spec[n].nx, spec[n].box, spec[n].dt, &spec[n].density, region->nx[1]);

		particles = &region->species[n].main_vector;

		const int ppc = region->species[n].ppc[1] * region->species[n].ppc[0];

		switch (spec[n].density.type)
		{
			case STEP:
				start = spec->density.start / spec->dx[0];
				particles->size = (region->nx[0] - start) * region->nx[1] * ppc;
				break;

			case SLAB:
				start = spec->density.start / spec->dx[0] - spec->n_move;
				end = spec->density.end / spec->dx[0] - spec->n_move;
				particles->size = (end - start) * region->nx[1] * ppc;
				break;

			default:
				particles->size = region->nx[0] * region->nx[1] * ppc;
				break;
		}

		memcpy(particles->part, spec[n].main_vector.part, particles->size * sizeof(t_part));

		spec[n].main_vector.size -= particles->size;
		void *restrict ptr = malloc(spec[n].main_vector.size * sizeof(t_part));

		if(ptr)
		{
			memcpy(ptr, spec[n].main_vector.part + particles->size, spec[n].main_vector.size * sizeof(t_part));
			free(spec[n].main_vector.part);
			spec[n].main_vector.part = ptr;
		}
	}

	//Calculate the region box
	float region_box[] = {box[0], box[1] / nx[1] * region->nx[1]};

	// Initialise the local current
	current_new(&region->local_current, region->nx, region_box, dt);

	// Initialise the local emf
	emf_new(&region->local_emf, region->nx, region_box, dt);

	// Initialise the others regions recursively
	if (id + 1 < n_regions)
	{
		region->next = malloc(sizeof(t_region));
		assert(region->next);

		region_new(region->next, n_regions, nx, id + 1, n_spec, spec, box, dt, gpu_percentage, n_gpu_regions, region);
	} else
	{
		t_region *restrict p = region;

		// Go to the first region
		while (p->prev != NULL && p->id != 0) p = p->prev;

		// Bridge the last region with the first
		p->prev = region;
		region->next = p;

		_n_regions = n_regions;
		_effective_gpu_regions = gpu_percentage * n_regions;
	}
}

// Link two adjacent regions and calculate the overlap zone between them
void region_link_adj_regions(t_region *region)
{
	current_overlap_zone(&region->local_current, &region->prev->local_current);
	emf_overlap_zone(&region->local_emf, &region->prev->local_emf);

	if(region->enable_gpu)
	{
		for(int n = 0; n < region->n_species; n++)
		{
			convert_vector(&region->species[n].main_vector, SoA);
			convert_vector(&region->species[n].temp_buffer[0], SoA);
			convert_vector(&region->species[n].temp_buffer[1], SoA);

			convert_vector(&region->prev->species[n].temp_buffer[1], SoA);
			convert_vector(&region->next->species[n].temp_buffer[0], SoA);
		}
	}
}

// Set moving window
void region_set_moving_window(t_region *region)
{
	region->local_current.moving_window = true;
	region->local_emf.moving_window = true;

	for (int i = 0; i < region->n_species; i++)
		region->species[i].moving_window = true;
}

// Add a laser to all the regions
void region_add_laser(t_region *region, t_emf_laser *laser)
{
	if (region->id != 0) while (region->id != 0) region = region->next;

	t_region *p = region;
	do
	{
		emf_add_laser(&p->local_emf, laser, p->limits_y[0]);
		p = p->next;
	} while (p->id != 0);

	p = region;
	do
	{
		if (p->id != 0) emf_update_gc_y(&p->local_emf);
		p = p->next;
	} while (p->id != 0);

	p = region;
	do
	{
		div_corr_x(&p->local_emf);
		p = p->next;
	} while (p->id != 0);

	p = region;
	do
	{
		emf_update_gc_y(&p->local_emf);
		emf_update_gc_x(&p->local_emf);
		p = p->next;
	} while (p->id != 0);
}

void region_delete(t_region *region)
{
	current_delete(&region->local_current);
	emf_delete(&region->local_emf);

	for (int i = 0; i < region->n_species; i++)
		spec_delete(&region->species[i]);
	free(region->species);
}

/*********************************************************************************************
 Advance
 *********************************************************************************************/
// Spec advance for all the regions (recursively)
void region_spec_advance(t_region *region)
{
	current_zero(&region->local_current);

	// Advance iteration count
	region->iter++;

	if (region->enable_gpu)
	{
		for (int i = 0; i < region->n_species; i++)
		{
			spec_advance_openacc(&region->species[i], &region->local_emf, &region->local_current,
					region->limits_y);
			spec_post_processing_1_openacc(&region->species[i], &region->next->species[i],
					&region->prev->species[i], region->limits_y);
		}
	} else
	{
		for (int i = 0; i < region->n_species; i++)
		{
			spec_advance(&region->species[i], &region->local_emf, &region->local_current,
					region->limits_y);
			spec_post_processing(&region->species[i], &region->next->species[i],
					&region->prev->species[i], region->limits_y);
		}
	}

	if (region->next->id != 0) region_spec_advance(region->next);
}

// Update the particle vector in all the regions (recursively)
void region_spec_update(t_region *region)
{
	if(region->enable_gpu)
	{
		for (int i = 0; i < region->n_species; i++)
		{
			spec_post_processing_2_openacc(&region->species[i], region->limits_y);
			if(region->iter % SORT_ITER == 0) spec_sort_openacc(&region->species[i], region->limits_y);
		}
	}

	else for (int i = 0; i < region->n_species; i++)
			spec_update_main_vector(&region->species[i]);

	if (region->next->id != 0) region_spec_update(region->next);
}

// Current reduction in y for all the regions (recursive calling)
void region_current_reduction_x(t_region *region)
{
	if(region->local_current.moving_window) return;

	if(region->enable_gpu) current_reduction_x_openacc(&region->local_current);
	else current_reduction_x(&region->local_current);
	if (region->next->id != 0) region_current_reduction_x(region->next);
}

// Current reduction in y for all the regions (recursive calling)
void region_current_reduction_y(t_region *region)
{
	if(region->enable_gpu) current_reduction_y_openacc(&region->local_current);
	else current_reduction_y(&region->local_current);
	if (region->next->id != 0) region_current_reduction_y(region->next);
}

// Apply the filter to the current buffer in all regions recursively. Then is necessary to
// update the ghost cells for produce correct results
void region_current_smooth(t_region *region, enum CURRENT_SMOOTH_MODE mode)
{
	switch (mode)
	{
		case SMOOTH_X:
			if(region->enable_gpu) current_smooth_x_openacc(&region->local_current);
			else current_smooth_x(&region->local_current);
			break;

		case CURRENT_UPDATE_GC:
			if(region->enable_gpu) current_gc_update_y_openacc(&region->local_current);
			else current_gc_update_y(&region->local_current);
			break;

		default:
			break;
	}

	if (region->next->id != 0) region_current_smooth(region->next, mode);
}

// Advance the EMF in each region recursively. Then is necessary update the ghost cells to reflect the
// new values
void region_emf_advance(t_region *region, enum EMF_UPDATE mode)
{
	switch (mode)
	{
		case EMF_ADVANCE:
			if(region->enable_gpu) emf_advance_openacc(&region->local_emf, &region->local_current);
			else emf_advance(&region->local_emf, &region->local_current);
			break;

		case EMF_UPDATE_GC:
			if(region->enable_gpu) emf_update_gc_y_openacc(&region->local_emf);
			else emf_update_gc_y(&region->local_emf);
			break;

		default:
			break;
	}

	if (region->next->id != 0) region_emf_advance(region->next, mode);
}

// Advance one iteration for all the regions. Always begin with the first region (id = 0)
void region_advance(t_region *region)
{
	if (region->id != 0) while (region->id != 0)
		region = region->next;

	region_spec_advance(region);
	region_spec_update(region);

	region_current_reduction_x(region);
	region_current_reduction_y(region);

	if (region->local_current.smooth.xtype != NONE)
	{
		region_current_smooth(region, SMOOTH_X);
		region_current_smooth(region, CURRENT_UPDATE_GC);
	}

	region_emf_advance(region, EMF_ADVANCE);
	region_emf_advance(region, EMF_UPDATE_GC);
}

/*********************************************************************************************
 Diagnostics
 *********************************************************************************************/
void region_charge_report(const t_region *region, t_part_data *charge, int i_spec)
{
	spec_deposit_charge(&region->species[i_spec], charge);
}

void region_emf_report(const t_region *region, t_fld *restrict E_mag, t_fld *restrict B_mag,
		const int nrow)
{
	emf_report_magnitude(&region->local_emf, E_mag, B_mag, nrow, region->limits_y[0]);
}
