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
#include <openacc.h>

#include "timer.h"
#include "utilities.h"

static int _num_gpus = 0;

/*********************************************************************************************
 Initialisation
 *********************************************************************************************/

void region_new(t_region *region, int n_regions, int nx[2], int id, int n_spec, t_species *spec,
		float box[], float dt, t_region *prev_region, t_region *next_region)
{
	region->id = id;
	region->prev = prev_region;
	region->next = next_region;

	region->iter = 0;

	region->limits_y[0] = floor((float) id * nx[1] / n_regions);
	region->limits_y[1] = floor((float) (id + 1) * nx[1] / n_regions);

	region->nx[0] = nx[0];
	region->nx[1] = region->limits_y[1] - region->limits_y[0];

	region->iter_time = 0.0;

	// Initialise particles in the region
	t_particle_vector *restrict particles;

	region->n_species = n_spec;
	region->species = (t_species*) malloc(n_spec * sizeof(t_species));
	assert(region->species);

	int start, end;
	for (int n = 0; n < n_spec; ++n)
	{
		spec_new(&region->species[n], spec[n].name, spec[n].m_q, spec[n].ppc, spec[n].ufl,
					spec[n].uth, spec[n].nx, spec[n].box, spec[n].dt, &spec[n].density,
					region->nx[1]);

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
		void *restrict ptr = alloc_align_buffer(DEFAULT_ALIGNMENT,
												spec[n].main_vector.size * sizeof(t_part));

		if (ptr)
		{
			memcpy(ptr, spec[n].main_vector.part + particles->size,
					spec[n].main_vector.size * sizeof(t_part));
			free_align_buffer(spec[n].main_vector.part);
			spec[n].main_vector.part = ptr;
		}
	}

	//Calculate the region box
	float region_box[] = {box[0], box[1] / nx[1] * region->nx[1]};

	// Initialise the local current
	current_new(&region->local_current, region->nx, region_box, dt);

	// Initialise the local emf
	emf_new(&region->local_emf, region->nx, region_box, dt);

	if(id == 0) _num_gpus = acc_get_num_devices(DEVICE_TYPE);
}

// Link two adjacent regions and calculate the overlap zone between them
void region_link_adj_regions(t_region *region)
{
	current_overlap_zone(&region->local_current, &region->prev->local_current);
	emf_overlap_zone(&region->local_emf, &region->prev->local_emf);

	for (int n = 0; n < region->n_species; n++)
	{
		convert_vector(&region->species[n].main_vector, SoA);
		convert_vector(&region->species[n].temp_buffer[0], SoA);
		convert_vector(&region->species[n].temp_buffer[1], SoA);
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
	if (region->id != 0) while (region->id != 0)
		region = region->next;

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

void region_advance(t_region *regions, const int n_regions)
{
	#pragma omp for schedule(static, 1)
	for(int i = 0; i < n_regions; i++)
	{
		acc_set_device_num(i % _num_gpus, DEVICE_TYPE);

		current_zero_openacc(&regions[i].local_current);

		for (int k = 0; k < regions[i].n_species; k++)
		{
			spec_sort_openacc(&regions[i].species[k], regions[i].limits_y);
			spec_advance_openacc(&regions[i].species[k], &regions[i].local_emf, &regions[i].local_current,
								 regions[i].limits_y);
			spec_post_processing_1_openacc(&regions[i].species[k], &regions[i].next->species[k],
											&regions[i].prev->species[k], regions[i].limits_y);
		}

		if(!regions[i].local_current.moving_window) current_reduction_x(&regions[i].local_current);
	}

	#pragma omp for schedule(static, 1)
	for(int i = 0; i < n_regions; i++)
	{
		// Advance iteration count
		regions[i].iter++;

		acc_set_device_num(i % _num_gpus, DEVICE_TYPE);

		for (int k = 0; k < regions[i].n_species; k++)
			spec_post_processing_2_openacc(&regions[i].species[k], regions[i].limits_y);
	}

	#pragma omp for schedule(static, 1)
	for(int i = 0; i < n_regions; i++)
	{
		acc_set_device_num(i % _num_gpus, DEVICE_TYPE);
		current_reduction_y_openacc(&regions[i].local_current);
	}

	if (regions[0].local_current.smooth.xtype != NONE)
	{
		#pragma omp for schedule(static, 1)
		for(int i = 0; i < n_regions; i++)
		{
			acc_set_device_num(i % _num_gpus, DEVICE_TYPE);
			current_smooth_x_openacc(&regions[i].local_current);
		}

		#pragma omp for schedule(static, 1)
		for(int i = 0; i < n_regions; i++)
		{
			acc_set_device_num(i % _num_gpus, DEVICE_TYPE);
			current_gc_update_y_openacc(&regions[i].local_current);
		}
	}

	if (regions[0].local_current.smooth.ytype != NONE)
	{
		for (int k = 0; k < regions[0].local_current.smooth.ylevel; k++)
		{
			#pragma omp for schedule(static, 1)
			for(int i = 0; i < n_regions; i++)
			{
				acc_set_device_num(i % _num_gpus, DEVICE_TYPE);
				current_smooth_y(&regions[i].local_current, BINOMIAL);
			}

			#pragma omp for schedule(static, 1)
			for(int i = 0; i < n_regions; i++)
			{
				acc_set_device_num(i % _num_gpus, DEVICE_TYPE);
				current_gc_update_y_openacc(&regions[i].local_current);
			}
		}

		if (regions[0].local_current.smooth.ytype == COMPENSATED)
		{
			#pragma omp for schedule(static, 1)
			for(int i = 0; i < n_regions; i++)
			{
				acc_set_device_num(i % _num_gpus, DEVICE_TYPE);
				current_smooth_y(&regions[i].local_current, COMPENSATED);
			}

			#pragma omp for schedule(static, 1)
			for(int i = 0; i < n_regions; i++)
			{
				acc_set_device_num(i % _num_gpus, DEVICE_TYPE);
				current_gc_update_y_openacc(&regions[i].local_current);
			}
		}
	}

	#pragma omp for schedule(static, 1)
	for(int i = 0; i < n_regions; i++)
	{
		acc_set_device_num(i % _num_gpus, DEVICE_TYPE);
		emf_advance_openacc(&regions[i].local_emf, &regions[i].local_current);
	}

	#pragma omp for schedule(static, 1)
	for(int i = 0; i < n_regions; i++)
	{
		acc_set_device_num(i % _num_gpus, DEVICE_TYPE);
		emf_update_gc_y_openacc(&regions[i].local_emf);
	}
}
