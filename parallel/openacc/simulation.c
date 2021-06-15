/*********************************************************************************************
 ZPIC
 simulation.c

 Created by Ricardo Fonseca on 11/8/10.
 Modified by Nicolas Guidotti on 11/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <assert.h>

#include "zdf.h"
#include "simulation.h"
#include "timer.h"
#include "utilities.h"

/*********************************************************************************************
 Initialisation
 *********************************************************************************************/

// Create output directory
void sim_create_dir(t_simulation *sim)
{
	char fullpath[256];

	//Create the output directory if it doesn't exists
	strcpy(fullpath, "output");

	struct stat sb;
	if (stat(fullpath, &sb) == -1)
	{
		mkdir(fullpath, 0700);
	}

	strcat(fullpath, "/");
	strcat(fullpath, sim->name);

	if (stat(fullpath, &sb) == -1)
	{
		mkdir(fullpath, 0700);
	}
}

// Constructor
void sim_new(t_simulation *sim, int nx[], float box[], float dt, float tmax, int ndump,
		t_species *species, int n_species, char name[64], int n_regions)
{
	// Simulation parameters
	sim->iter = 0;
	sim->moving_window = false;
	sim->dt = dt;
	sim->tmax = tmax;
	sim->ndump = ndump;
	sim->nx[0] = nx[0];
	sim->nx[1] = nx[1];
	sim->box[0] = box[0];
	sim->box[1] = box[1];
	strncpy(sim->name, name, 64);

	// Check time step
	float dx[] = { box[0] / nx[0], box[1] / nx[1] };
	float cour = sqrtf(1.0f / (1.0f / (dx[0] * dx[0]) + 1.0f / (dx[1] * dx[1])));
	if (dt >= cour)
	{
		fprintf(stderr, "Invalid timestep, courant condition violation, dtmax = %f \n", cour);
		exit(-1);
	}

	// Inject particles in the simulation that will be distributed to all the regions
	const int range[][2] = { { 0, nx[0] }, { 0, nx[1] } };
	for (int n = 0; n < n_species; ++n)
		spec_inject_particles(&species[n].main_vector, range, species[n].ppc, &species[n].density,
				species[n].dx, species[n].n_move, species[n].ufl, species[n].uth);

	// Initialise the regions
	sim->n_regions = n_regions;
	sim->regions = malloc(n_regions * sizeof(t_region));
	assert(sim->regions);

	t_region *prev = &sim->regions[n_regions - 1];
	for(int i = 0; i < n_regions; i++)
	{
		t_region *next = i == n_regions - 1 ? &sim->regions[0] : &sim->regions[i + 1];
		region_new(&sim->regions[i], n_regions, nx, i, n_species, species, box, dt, prev, next);
		prev = &sim->regions[i];
	}

	for(int i = 0; i < n_regions; i++)
		region_init(&sim->regions[i]);

	// Cleaning
	for (int n = 0; n < n_species; ++n)
		spec_delete(&species[n]);

	sim_create_dir(sim);

	char filename[128];
	FILE *file;
	sprintf(filename, "output/%s/energy.csv", sim->name);
	file = fopen(filename, "w+");
	fclose(file);
}

void sim_delete(t_simulation *sim)
{
	for(int i = 0; i < sim->n_regions; i++)
		region_delete(&sim->regions[i]);
	free(sim->regions);
}

void sim_add_laser(t_simulation *sim, t_emf_laser *laser)
{
	for(int i = 0; i < sim->n_regions; i++)
		emf_add_laser(&sim->regions[i].local_emf, laser, sim->regions[i].limits_y[0]);

	for(int i = 1; i < sim->n_regions; i++)
		emf_update_gc_y(&sim->regions[i].local_emf);

	for(int i = 0; i < sim->n_regions; i++)
		emf_update_gc_x(&sim->regions[i].local_emf);

	for(int i = 0; i < sim->n_regions; i++)
		div_corr_x(&sim->regions[i].local_emf);

	for(int i = 0; i < sim->n_regions; i++)
		emf_update_gc_y(&sim->regions[i].local_emf);

	for(int i = 0; i < sim->n_regions; i++)
		emf_update_gc_x(&sim->regions[i].local_emf);
}

void sim_set_smooth(t_simulation *sim, t_smooth *smooth)
{
	if ((smooth->xtype != NONE) && (smooth->xlevel <= 0))
	{
		fprintf(stderr, "Invalid smooth level along x direction\n");
		exit(-1);
	}

	if ((smooth->ytype != NONE) && (smooth->ylevel <= 0))
	{
		fprintf(stderr, "Invalid smooth level along y direction\n");
		exit(-1);
	}

	for(int i = 0; i < sim->n_regions; i++)
		sim->regions[i].local_current.smooth = *smooth;

}

void sim_set_moving_window(t_simulation *sim)
{
	sim->moving_window = true;

	for(int i = 0; i < sim->n_regions; i++)
		region_set_moving_window(&sim->regions[i]);
}

/*********************************************************************************************
 Iteration
 *********************************************************************************************/
void sim_iter(t_simulation *sim)
{
	const int num_gpus = acc_get_num_devices(DEVICE_TYPE);

	t_region *regions = sim->regions;
	const int n_regions = sim->n_regions;

	#pragma omp for schedule(static, 1)
	for(int i = 0; i < n_regions; i++)
	{
		const int device = i % num_gpus;
		acc_set_device_num(device, DEVICE_TYPE);

		// Advance iteration count
		regions[i].iter++;

		current_zero_openacc(&regions[i].local_current, device);

		for (int k = 0; k < regions[i].n_species; k++)
		{
			spec_advance_openacc(&regions[i].species[k], &regions[i].local_emf,
					&regions[i].local_current, regions[i].limits_y, device);
			if (regions[i].species[k].moving_window)
				spec_move_window_openacc(&regions[i].species[k], regions[i].limits_y, device);
			spec_check_boundaries_openacc(&regions[i].species[k], regions[i].limits_y, device);
		}

		if (!regions[i].local_current.moving_window)
			current_reduction_x_openacc(&regions[i].local_current, device);
	}

	#pragma omp for schedule(static, 1)
	for(int i = 0; i < n_regions; i++)
	{
		const int device = i % num_gpus;
		acc_set_device_num(i % num_gpus, DEVICE_TYPE);

		for (int k = 0; k < regions[i].n_species; k++)
			spec_sort_openacc(&regions[i].species[k], regions[i].limits_y, device);
		current_reduction_y_openacc(&regions[i].local_current, device);
	}

	if (regions[0].local_current.smooth.xtype != NONE)
	{
		#pragma omp for schedule(static, 1)
		for(int i = 0; i < n_regions; i++)
		{
			const int device = i % num_gpus;
			acc_set_device_num(i % num_gpus, DEVICE_TYPE);
			current_smooth_x_openacc(&regions[i].local_current, device);
		}
	}

	if (regions[0].local_current.smooth.ytype != NONE)
	{
		for (int k = 0; k < regions[0].local_current.smooth.ylevel; k++)
		{
			#pragma omp for schedule(static, 1)
			for(int i = 0; i < n_regions; i++)
				current_smooth_y(&regions[i].local_current, BINOMIAL);

			#pragma omp for schedule(static, 1)
			for(int i = 0; i < n_regions; i++)
			{
				const int device = i % num_gpus;
				acc_set_device_num(device, DEVICE_TYPE);
				current_gc_update_y_openacc(&regions[i].local_current, device);
			}
		}

		if (regions[0].local_current.smooth.ytype == COMPENSATED)
		{
			#pragma omp for schedule(static, 1)
			for(int i = 0; i < n_regions; i++)
				current_smooth_y(&regions[i].local_current, COMPENSATED);

			#pragma omp for schedule(static, 1)
			for(int i = 0; i < n_regions; i++)
			{
				const int device = i % num_gpus;

				acc_set_device_num(device, DEVICE_TYPE);
				current_gc_update_y_openacc(&regions[i].local_current, device);
			}
		}
	}

	#pragma omp for schedule(static, 1)
	for(int i = 0; i < n_regions; i++)
	{
		const int device = i % num_gpus;
		acc_set_device_num(device, DEVICE_TYPE);
		emf_advance_openacc(&regions[i].local_emf, &regions[i].local_current, device);
	}

	#pragma omp for schedule(static, 1)
	for(int i = 0; i < n_regions; i++)
	{
		const int device = i % num_gpus;
		acc_set_device_num(device, DEVICE_TYPE);
		emf_update_gc_y_openacc(&regions[i].local_emf, device);
	}
}

/*********************************************************************************************
 Diagnostics
 *********************************************************************************************/
int report(int n, int ndump)
{
	if (ndump > 0)
	{
		return !(n % ndump);
	} else
	{
		return 0;
	}
}

// Save the simulation energy to a CSV file
void sim_report_energy(t_simulation *sim)
{
	char filename[128];

	double tot_emf = 0;
	double tot_part = 0;

	for(int i = 0; i < sim->n_regions; i++)
	{
		tot_emf += emf_get_energy(&sim->regions[i].local_emf);

		for (int k = 0; k < sim->regions[i].n_species; k++)
		{
			spec_calculate_energy(&sim->regions[i].species[k]);
			tot_part += sim->regions[i].species[k].energy;
		}
	}

	sprintf(filename, "output/%s/energy.csv", sim->name);
	FILE *file = fopen(filename, "a+");

	if (file)
	{
		fprintf(file, "%e;%e;%e\n", tot_emf, tot_part, tot_emf + tot_part);
		fclose(file);

	} else
	{
		printf("Error on open file: %s", filename);
		exit(1);
	}
}

void sim_timings(t_simulation *sim, uint64_t t0, uint64_t t1, const unsigned int n_iterations)
{
	int npart = 0;
	int n_regions = 0;

	t_region *restrict region = &sim->regions[0];
	do
	{
		for (int i = 0; i < sim->regions[0].n_species; i++)
			npart += region->species[i].main_vector.size;

		region = region->next;
		n_regions++;
	} while (region->id != 0);

#ifndef TEST
	fprintf(stdout, "Simulation: %s\n", sim->name);
	fprintf(stdout, "Number of regions (Total): %d\n", sim->n_regions);
	fprintf(stdout, "Number of GPUs: %d\n", acc_get_num_devices(DEVICE_TYPE));
	fprintf(stdout, "Sort - Bin size: %d\n", TILE_SIZE);

#ifdef ENABLE_PREFETCH
	fprintf(stdout, "Prefetch: Enable\n");
#else
	fprintf(stdout, "Prefetch: Disable\n");
#endif

	fprintf(stdout, "Total simulation time  = %f s\n", timer_interval_seconds(t0, t1));
	fprintf(stdout, "Time per iteration = %f ms\n", timer_interval_seconds(t0, t1) / n_iterations * 1E3);
	fprintf(stdout, "\n");

#else
#ifdef ENABLE_PREFETCH
	printf("%s,%d,%d,%d,1,%f\n", sim->name, sim->n_regions, acc_get_num_devices(DEVICE_TYPE), TILE_SIZE, timer_interval_seconds(t0, t1));
#else
	printf("%s,%d,%d,%d,0,%f\n", sim->name, sim->n_regions, acc_get_num_devices(DEVICE_TYPE), TILE_SIZE, timer_interval_seconds(t0, t1));
#endif
#endif
}

// Save the grid quantity to a ZDF file
void sim_report_grid_zdf(t_simulation *sim, enum report_grid_type type, const int coord)
{
	t_region *restrict region = &sim->regions[0];
	t_fld *restrict global_buf = calloc(sim->nx[0] * sim->nx[1], sizeof(t_fld));
	char path[128] = "";
	sprintf(path, "output/%s/grid", sim->name);

	switch (type)
	{
		case REPORT_BFLD:
			do
			{
				emf_reconstruct_global_buffer(&region->local_emf, global_buf, region->limits_y[0],
						BFLD, coord);
				region = region->next;
			} while (region->id != 0);

			emf_report(global_buf, sim->box, sim->nx, sim->iter, sim->dt, BFLD, coord, path);
			break;

		case REPORT_EFLD:
			do
			{
				emf_reconstruct_global_buffer(&region->local_emf, global_buf, region->limits_y[0],
						EFLD, coord);
				region = region->next;
			} while (region->id != 0);

			emf_report(global_buf, sim->box, sim->nx, sim->iter, sim->dt, EFLD, coord, path);
			break;

		case REPORT_CURRENT:
			do
			{
				current_reconstruct_global_buffer(&region->local_current, global_buf,
						region->limits_y[0], coord);
				region = region->next;
			} while (region->id != 0);

			current_report(global_buf, sim->iter, sim->nx, sim->box, sim->dt, coord, path);
			break;

		default:
			break;
	}

	free(global_buf);
}

// Save a particle property to a ZDF file
void sim_report_spec_zdf(t_simulation *sim, const int species, const int rep_type,
		const int pha_nx[2], const float pha_range[][2])
{
	size_t size;
	t_region *restrict region = &sim->regions[0];
	char path[128] = "";
	sprintf(path, "output/%s/%s", sim->name, sim->regions[0].species[species].name);

	switch (rep_type & 0xF000)
	{
		case CHARGE:
		{
			size = (sim->nx[0] + 1) * (sim->nx[1] + 1) * sizeof(t_part_data);  // Add 1 guard cell to the upper boundary
			t_part_data *restrict charge = malloc(size);
			memset(charge, 0, size);

			region = &sim->regions[0];
			do
			{
				spec_deposit_charge(&region->species[species], charge);
				region = region->next;
			} while (region->id != 0);

			spec_rep_charge(charge, sim->nx, sim->box, sim->iter, sim->dt, sim->moving_window,
							path);

			free(charge);
		}
			break;

		case PHA:
		{
			float *buf = malloc(pha_nx[0] * pha_nx[1] * sizeof(float));
			memset(buf, 0, pha_nx[0] * pha_nx[1] * sizeof(float));

			region = &sim->regions[0];
			do
			{
				spec_deposit_pha(&region->species[species], rep_type, pha_nx, pha_range, buf);
				region = region->next;
			} while (region->id != 0);

			spec_rep_pha(buf, rep_type, pha_nx, pha_range, sim->iter, sim->dt, path);

			free(buf);

		}
			break;

		case PARTICLES:
		{
			const char *quants[] = {"x1", "x2", "u1", "u2", "u3"};
			const char *units[] = {"c/\\omega_p", "c/\\omega_p", "c", "c", "c"};

			t_zdf_iteration iter = {.n = sim->iter, .t = sim->iter * sim->dt,
					.time_units = "1/\\omega_p"};

			// Allocate buffer for positions
			int np = 0;
			region = &sim->regions[0];
			do
			{
				np += region->species[species].main_vector.size;
				region = region->next;
			} while (region->id != 0);

			size = np * sizeof(float);
			float *data = malloc(size);

			t_zdf_part_info info = {.name = (char*) sim->name, .nquants = 5,
					.quants = (char**) quants, .units = (char**) units, .np = np};

			// Create file and add description
			t_zdf_file part_file;
			zdf_part_file_open(&part_file, &info, &iter, path);

			// Add positions and generalized velocities
			t_species *restrict spec;
			int offset;

			// x1
			region = &sim->regions[0];
			offset = 0;
			do
			{
				spec = &region->species[species];

				for (int i = 0; i < spec->main_vector.size; i++)
					data[i + offset] = (spec->n_move + spec->main_vector.ix[i]
							+ spec->main_vector.x[i]) * spec->dx[0];

				region = region->next;
				offset += spec->main_vector.size;
			} while (region->id != 0);
			zdf_part_file_add_quant(&part_file, quants[0], data, np);

			// x2
			region = &sim->regions[0];
			offset = 0;
			do
			{
				spec = &region->species[species];

				for (int i = 0; i < spec->main_vector.size; i++)
					data[i + offset] = (spec->n_move + spec->main_vector.iy[i]
							+ spec->main_vector.y[i]) * spec->dx[1];

				region = region->next;
				offset += spec->main_vector.size;
			} while (region->id != 0);
			zdf_part_file_add_quant(&part_file, quants[1], data, np);

			// ux
			region = &sim->regions[0];
			offset = 0;
			do
			{
				spec = &region->species[species];

				for (int i = 0; i < spec->main_vector.size; i++)
					data[i + offset] = spec->main_vector.ux[i];

				region = region->next;
				offset += spec->main_vector.size;
			} while (region->id != 0);
			zdf_part_file_add_quant(&part_file, quants[2], data, np);

			// uy
			region = &sim->regions[0];
			offset = 0;
			do
			{
				spec = &region->species[species];

				for (int i = 0; i < spec->main_vector.size; i++)
					data[i + offset] = spec->main_vector.uy[i];

				region = region->next;
				offset += spec->main_vector.size;
			} while (region->id != 0);
			zdf_part_file_add_quant(&part_file, quants[3], data, np);

			// uz
			region = &sim->regions[0];
			offset = 0;
			do
			{
				spec = &region->species[species];

				for (int i = 0; i < spec->main_vector.size; i++)
					data[i + offset] = spec->main_vector.uz[i];

				region = region->next;
				offset += spec->main_vector.size;
			} while (region->id != 0);
			zdf_part_file_add_quant(&part_file, quants[4], data, np);

			free(data);
			zdf_close_file(&part_file);

		}
			break;
		default:
			break;
	}
}

