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

#include "utilities.h"

/*********************************************************************************************
 Initialisation
 *********************************************************************************************/

// Build a double-linked list of the regions recursively
void region_new(t_region *region, int id, int n_regions, int proc_nx[2], int proc_limits[2][2],
                float proc_box[], int n_spec, t_species *spec, float dt, bool on_right_edge,
                bool on_left_edge, t_region *prev_region, t_region *next_region)
{
	region->id = id;
	region->prev = prev_region; // Previous region in the list
	region->next = next_region;

	// Region boundaries
	region->limits[0][0] = proc_limits[0][0];
	region->limits[0][1] = proc_limits[0][1];
	region->limits[1][0] = proc_limits[1][0] + floor((float) id * proc_nx[1] / n_regions);
	region->limits[1][1] = proc_limits[1][0] + floor((float) (id + 1) * proc_nx[1] / n_regions);
	region->nx[0] = proc_nx[0];
	region->nx[1] = region->limits[1][1] - region->limits[1][0];

	//Calculate the region box
	float region_box[] = {proc_box[0], proc_box[1] / proc_nx[1] * region->nx[1]};

	// Initialise the particles inside the region
	t_part_vector *restrict particles;

	region->n_species = n_spec;
	region->species = (t_species*) malloc(n_spec * sizeof(t_species));
	assert(region->species);

	int start, end;
	for (int n = 0; n < n_spec; ++n)
	{
		spec_new(&region->species[n], spec[n].name, spec[n].m_q, spec[n].ppc, spec[n].ufl,
				spec[n].uth, region->nx, region_box, spec[n].dt, &spec[n].density);

		particles = &region->species[n].main_vector;

		const int ppc = region->species[n].ppc[1] * region->species[n].ppc[0];

		switch (spec[n].density.type)
		{
			case STEP:
				start = spec->density.start / spec->dx[0];
				start = MAX_VALUE(start, region->limits[0][0]);

				particles->size = (region->limits[0][1] - start) * region->nx[1] * ppc;
				break;

			case SLAB:
				start = spec->density.start / spec->dx[0];
				start = MAX_VALUE(start, region->limits[0][0]);

				end = spec->density.end / spec->dx[0];
				end = MIN_VALUE(end, region->limits[0][1]);

				particles->size = (end - start) * region->nx[1] * ppc;
				break;

			default:
				particles->size = region->nx[0] * region->nx[1] * ppc;
				break;
		}

		if(particles->size < 0) particles->size = 0;

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

	// Initialise the local current
	current_new(&region->local_current, region->nx, region_box, dt, on_right_edge, on_left_edge);

	// Initialise the local emf
	emf_new(&region->local_emf, region->nx, region_box, dt, on_right_edge, on_left_edge);
}

// Link the outgoing and incoming buffer from adjacent regions
void region_link_adj_part(t_region *region, t_part **gaspi_segm_part,
						  const int **gaspi_segm_offset, const int proc_limits[2][2],
						  const int sim_nx[2])
{
	for (int n = 0; n < region->n_species; n++)
	{
		t_part_vector *adj_spec[8];

		for (int i = 0; i < 8; ++i)
			adj_spec[i] = NULL;

		if (region->prev)
		{
			adj_spec[PART_DOWN_LEFT] = &region->prev->species[n].incoming_part[PART_UP_LEFT];
			adj_spec[PART_DOWN] = &region->prev->species[n].incoming_part[PART_UP];
			adj_spec[PART_DOWN_RIGHT] = &region->prev->species[n].incoming_part[PART_UP_RIGHT];
		}

		if (region->next)
		{
			adj_spec[PART_UP_LEFT] = &region->next->species[n].incoming_part[PART_DOWN_LEFT];
			adj_spec[PART_UP] = &region->next->species[n].incoming_part[PART_DOWN];
			adj_spec[PART_UP_RIGHT] = &region->next->species[n].incoming_part[PART_DOWN_RIGHT];
		}

		spec_link_adj_regions(&region->species[n], adj_spec, gaspi_segm_part[n],
							  gaspi_segm_offset[n], n, region->id, region->nx,
							  region->limits, proc_limits);
	}
}

// Link the grid between adjacent regions
void region_link_adj_grid(t_region *region, t_vfld *gaspi_segm_J, t_vfld *gaspi_segm_E,
						  t_vfld *gaspi_segm_B, const int emf_segm_offset[8],
						  const int current_segm_offset[8], const gaspi_rank_t adj_ranks[4],
						  const int proc_limits[2][2], const int sim_nx[2])
{
	t_current *current_down = region->prev ? &region->prev->local_current : NULL;
	t_current *current_up = region->next ? &region->next->local_current : NULL;
	current_link_adj_regions(&region->local_current, current_down, current_up, gaspi_segm_J,
							 current_segm_offset, adj_ranks, region->id, region->limits,
							 proc_limits);

	t_emf *emf_down = region->prev ? &region->prev->local_emf : NULL;
	t_emf *emf_up = region->next ? &region->next->local_emf : NULL;
	emf_link_adj_regions(&region->local_emf, emf_down, emf_up, gaspi_segm_E, gaspi_segm_B,
						 emf_segm_offset, adj_ranks, region->id, region->limits, proc_limits);
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
