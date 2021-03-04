/*********************************************************************************************
 ZPIC
 region.c

 Created by Nicolas Guidotti on 11/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#include "region.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "timer.h"

static int _n_regions = 0;

int get_n_regions()
{
	return _n_regions;
}

/*********************************************************************************************
 Initialisation
 *********************************************************************************************/

// Build a double-linked list of the regions recursively
void region_new(t_region *region, int n_regions, int nx[2], int id, int n_spec, t_species *spec,
		float box[], float dt, t_region *prev_region)
{
	region->id = id;
	region->prev = prev_region; // Previous region in the list

	// Region boundaries
	region->limits_y[0] = floor((float) id * nx[1] / n_regions);
	region->limits_y[1] = floor((float) (id + 1) * nx[1] / n_regions);
	region->nx[0] = nx[0];
	region->nx[1] = region->limits_y[1] - region->limits_y[0];

	// Reset the region timer
	region->iter_time = 0.0;

	// Initialise the particles inside the region
	t_particle_vector *restrict particles;

	region->n_species = n_spec;
	region->species = (t_species*) malloc(n_spec * sizeof(t_species));
	assert(region->species);

	int start, end;
	for (int n = 0; n < n_spec; ++n)
	{
		spec_new(&region->species[n], spec[n].name, spec[n].m_q, spec[n].ppc, spec[n].ufl,
				spec[n].uth, spec[n].nx, spec[n].box, spec[n].dt, &spec[n].density);

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

		particles->size_max = particles->size;
		particles->data = malloc(particles->size * sizeof(t_part));
		memcpy(particles->data, spec[n].main_vector.data, particles->size * sizeof(t_part));

		spec[n].main_vector.size -= particles->size;
		void *restrict ptr = malloc(spec[n].main_vector.size * sizeof(t_part));

		if(ptr)
		{
			memcpy(ptr, spec[n].main_vector.data + particles->size, spec[n].main_vector.size * sizeof(t_part));
			free(spec[n].main_vector.data);
			spec[n].main_vector.data = ptr;
		}
	}

	//Calculate the region box
	float region_box[] = {box[0], box[1] / nx[1] * (region->limits_y[1] - region->limits_y[0])};

	// Initialise the local current
	current_new(&region->local_current, region->nx, region_box, dt);

	// Initialise the local emf
	emf_new(&region->local_emf, region->nx, region_box, dt);

	// Initialise the others regions recursively
	if (id + 1 < n_regions)
	{
		region->next = malloc(sizeof(t_region));
		assert(region->next);

		region_new(region->next, n_regions, nx, id + 1, n_spec, spec, box, dt, region);
	} else
	{
		t_region *restrict p = region;

		// Go to the first region
		while (p->id != 0) p = p->prev;

		// Bridge the last region with the first
		p->prev = region;
		region->next = p;

		_n_regions = n_regions;
	}
}

// Link two adjacent regions and calculate the overlap zone between them
void region_link_adj_regions(t_region *region)
{
	current_overlap_zone(&region->local_current, &region->prev->local_current);
	emf_overlap_zone(&region->local_emf, &region->prev->local_emf);

	for (int n = 0; n < region->n_species; n++)
	{
		region->species[n].outgoing_part[0] = &region->prev->species[n].incoming_part[1];
		region->species[n].outgoing_part[1] = &region->next->species[n].incoming_part[0];
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
	if (region->id != 0)
		while (region->id != 0) region = region->next;

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

	#pragma oss taskwait

	p = region;
	do
	{
		div_corr_x(&p->local_emf);
		emf_update_gc_x(&p->local_emf);
		p = p->next;
	} while (p->id != 0);

	p = region;
	do
	{
		emf_update_gc_y(&p->local_emf);
		p = p->next;
	} while (p->id != 0);

	#pragma oss taskwait
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

	for (int i = 0; i < region->n_species; i++)
		spec_advance(&region->species[i], &region->local_emf, &region->local_current,
				region->limits_y);

	if (region->next->id != 0) region_spec_advance(region->next);
}

// Update the particle vector in all the regions (recursively)
void region_spec_update(t_region *region)
{
	for (int i = 0; i < region->n_species; i++)
		spec_merge_vectors(&region->species[i]);

	if (region->next->id != 0) region_spec_update(region->next);
}

// Current reduction in y for all the regions (recursive calling)
void region_current_reduction_x(t_region *region)
{
	current_reduction_x(&region->local_current);
	if (region->next->id != 0) region_current_reduction_x(region->next);
}

// Current reduction in y for all the regions (recursive calling)
void region_current_reduction_y(t_region *region)
{
	current_reduction_y(&region->local_current);
	if (region->next->id != 0) region_current_reduction_y(region->next);
}

// Apply the filter to the current buffer in all regions recursively. To produce
// accurate results, the ghost cells MUST be updated after
void region_current_smooth(t_region *region, enum CURRENT_SMOOTH_MODE mode)
{
	switch (mode)
	{
		case SMOOTH_X:
			current_smooth_x(&region->local_current);
			break;

		case CURRENT_UPDATE_GC:
			current_gc_update_y(&region->local_current);
			break;

		case BINOMIAL_Y:
			current_smooth_y(&region->local_current, BINOMIAL);
			break;

		case COMPENSATED_Y:
			current_smooth_y(&region->local_current, COMPENSATED);
			break;

		default:
			break;
	}

	if (region->next->id != 0) region_current_smooth(region->next, mode);
}

// Advance the EMF in each region recursively. To produce
// accurate results, the ghost cells MUST be updated after
void region_emf_advance(t_region *region, enum EMF_UPDATE mode)
{
	switch (mode)
	{
		case EMF_ADVANCE:
			emf_advance(&region->local_emf, &region->local_current);
			break;

		case EMF_UPDATE_GC:
			emf_update_gc_y(&region->local_emf);
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

	// Particle
	region_spec_advance(region);
	region_spec_update(region);

	// Current
	region_current_reduction_x(region);
	region_current_reduction_y(region);

	if (region->local_current.smooth.xtype != NONE)
	{
		region_current_smooth(region, SMOOTH_X);
		region_current_smooth(region, CURRENT_UPDATE_GC);
	}

	if (region->local_current.smooth.ytype != NONE)
	{
		for(int i = 0; i < region->local_current.smooth.ylevel; i++)
		{
			region_current_smooth(region, BINOMIAL_Y);
			region_current_smooth(region, CURRENT_UPDATE_GC);
		}

		if(region->local_current.smooth.ytype == COMPENSATED)
		{
			region_current_smooth(region, COMPENSATED_Y);
			region_current_smooth(region, CURRENT_UPDATE_GC);
		}
	}

	// EMF
	region_emf_advance(region, EMF_ADVANCE);
	region_emf_advance(region, EMF_UPDATE_GC);
}
