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
#include "utilities.h"

/*********************************************************************************************
 Initialisation
 *********************************************************************************************/

void region_new(t_region *region, int n_regions, int nx[2], int id, int n_spec, t_species *spec,
		float box[], float dt, t_region *prev_region, t_region *next_region)
{
	const int num_gpus = acc_get_num_devices(DEVICE_TYPE);

	region->id = id;
	region->prev = prev_region;
	region->next = next_region;

	region->limits_y[0] = floor((float) id * nx[1] / n_regions);
	region->limits_y[1] = floor((float) (id + 1) * nx[1] / n_regions);

	region->nx[0] = nx[0];
	region->nx[1] = region->limits_y[1] - region->limits_y[0];

	// Initialise particles in the region
	t_particle_vector *restrict particles;

	region->n_species = n_spec;
	region->species = (t_species*) malloc(n_spec * sizeof(t_species));
	assert(region->species);

	int start, end;
	for (int n = 0; n < n_spec; ++n)
	{
		spec_new(&region->species[n], spec[n].name, spec[n].m_q, spec[n].ppc, spec[n].ufl,
				spec[n].uth, spec[n].nx, spec[n].box, spec[n].dt, &spec[n].density, region->nx[1], id);

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

		part_vector_memcpy(&spec[n].main_vector, particles, 0, particles->size);
		spec[n].main_vector.size -= particles->size;

		if(spec[n].main_vector.size > 0)
		{
			t_particle_vector tmp;
			part_vector_alloc(&tmp, spec[n].main_vector.size, 0);
			part_vector_memcpy(&spec[n].main_vector, &tmp, particles->size, spec[n].main_vector.size);
			part_vector_free(&spec[n].main_vector);

			spec[n].main_vector.ix = tmp.ix;
			spec[n].main_vector.iy = tmp.iy;
			spec[n].main_vector.x = tmp.x;
			spec[n].main_vector.y = tmp.y;
			spec[n].main_vector.ux = tmp.ux;
			spec[n].main_vector.uy = tmp.uy;
			spec[n].main_vector.uz = tmp.uz;
			spec[n].main_vector.invalid = tmp.invalid;
		}

#ifdef ENABLE_ADVISE
		part_vector_mem_advise(&region->species[n].main_vector,
		        CU_MEM_ADVISE_SET_PREFERRED_LOCATION, region->id % num_gpus);
#endif
	}

	//Calculate the region box
	float region_box[] = {box[0], box[1] / nx[1] * region->nx[1]};

	// Initialise the local current
	current_new(&region->local_current, region->nx, region_box, dt, id);

	// Initialise the local emf
	emf_new(&region->local_emf, region->nx, region_box, dt, id);

#ifdef ENABLE_ADVISE
	cuMemAdvise(region->local_current.J_buf, region->local_current.total_size,
	        CU_MEM_ADVISE_SET_PREFERRED_LOCATION, id % num_gpus);
	cuMemAdvise(region->local_emf.E_buf, region->local_emf.total_size,
	        CU_MEM_ADVISE_SET_PREFERRED_LOCATION, id % num_gpus);
	cuMemAdvise(region->local_emf.B_buf, region->local_emf.total_size,
	        CU_MEM_ADVISE_SET_PREFERRED_LOCATION, id % num_gpus);
#endif
}

// Link two adjacent regions and calculate the overlap zone between them
void region_init(t_region *region)
{
	const int num_gpus = acc_get_num_devices(DEVICE_TYPE);

	current_overlap_zone(&region->local_current, &region->prev->local_current, region->id);
	emf_overlap_zone(&region->local_emf, &region->prev->local_emf, region->id);

	for(int i = 0; i < region->n_species; i++)
	{
		region->species[i].outgoing_part[0] = &region->prev->species[i].incoming_part[1];
		region->species[i].outgoing_part[1] = &region->next->species[i].incoming_part[0];

		spec_organize_in_tiles(&region->species[i], region->limits_y, region->id);
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
