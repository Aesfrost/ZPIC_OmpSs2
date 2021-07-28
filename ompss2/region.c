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

/*********************************************************************************************
 Initialisation
 *********************************************************************************************/

// Initialize a given region
void region_new(t_region *region, int n_regions, int nx[2], int id, int n_spec, t_species *spec,
		float box[], float dt, t_region *prev_region, t_region *next_region)
{
	region->id = id;
	region->prev = prev_region; // Previous region in the list
	region->next = next_region;

	// Region boundaries
	region->limits_y[0] = floor((float) id * nx[1] / n_regions);
	region->limits_y[1] = floor((float) (id + 1) * nx[1] / n_regions);
	region->nx[0] = nx[0];
	region->nx[1] = region->limits_y[1] - region->limits_y[0];

	// Initialise the particles inside the region
	t_part_vector *restrict particles;

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

void region_delete(t_region *region)
{
	current_delete(&region->local_current);
	emf_delete(&region->local_emf);

	for (int i = 0; i < region->n_species; i++)
		spec_delete(&region->species[i]);
	free(region->species);
}
