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

#include "utilities.h"
#include "simulation.h"
#include "timer.h"
#include "zdf.h"

#ifdef ENABLE_TASKING
#include <nanos6.h>
#include "task_management.h"
#endif

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
void sim_new(t_simulation *sim, int nx[2], float box[2], float dt, float tmax, int ndump,
             t_species *species, int n_species, char name[64], int n_regions)
{
//	#pragma acc set device_num(0) // Dummy operation to work with the PGI Compiler

	CHECK_MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &sim->proc_rank));
	CHECK_MPI_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &sim->num_procs));

	get_optimal_division(sim->num_procs_cart, sim->num_procs);
//	sim->num_procs_cart[0] = 1;
//	sim->num_procs_cart[1] = sim->num_procs;

	sim->proc_rank_cart[0] = sim->proc_rank % sim->num_procs_cart[0];
	sim->proc_rank_cart[1] = sim->proc_rank / sim->num_procs_cart[0];

	// Calculate the rank of neighbour processes (all)
	int dir = 0;
	for (int j = -1; j <= 1; j++)
	{
		for (int i = -1; i <= 1; i++)
		{
			if (i != 0 || j != 0)
			{
				int x = PERIODIC_BOUNDARIES((int) sim->proc_rank_cart[0] + i, sim->num_procs_cart[0]);
				int y = PERIODIC_BOUNDARIES((int) sim->proc_rank_cart[1] + j, sim->num_procs_cart[1]);
				sim->adj_ranks_part[dir++] = x + y * sim->num_procs_cart[0];
			}
		}
	}

	// For the grid segments (E, B and J), consider only UP, DOWN, RIGHT and LEFT directions
	sim->adj_ranks_grid[GRID_UP] = sim->adj_ranks_part[PART_UP];
	sim->adj_ranks_grid[GRID_LEFT] = sim->adj_ranks_part[PART_LEFT];
	sim->adj_ranks_grid[GRID_RIGHT] = sim->adj_ranks_part[PART_RIGHT];
	sim->adj_ranks_grid[GRID_DOWN] = sim->adj_ranks_part[PART_DOWN];

	// Simulation parameters
	strncpy(sim->name, name, 64);
	sim->iter = 0;
	sim->moving_window = false;
	sim->dt = dt;
	sim->tmax = tmax;
	sim->ndump = ndump;

	// Determine if the process is in the left or right edge of the simulation
	sim->on_left_edge = (sim->proc_rank_cart[0] == 0);
	sim->on_right_edge = (sim->proc_rank_cart[0] == sim->num_procs_cart[0] - 1);

	for (int i = 0; i < 2; ++i)
	{
		sim->nx[i] = nx[i];
		sim->box[i] = box[i];
		sim->gc[i][0] = 1;
		sim->gc[i][1] = 2;

		sim->proc_limits[i][0] = floor((float) sim->proc_rank_cart[i] * nx[i] / sim->num_procs_cart[i]);
		sim->proc_limits[i][1] = floor((float) (sim->proc_rank_cart[i] + 1) * nx[i] / sim->num_procs_cart[i]);

		sim->proc_nx[i] = sim->proc_limits[i][1] - sim->proc_limits[i][0];
		sim->proc_box[i] = box[i] / nx[i] * sim->proc_nx[i];
	}

	// Check time step
	float dx[] = {box[0] / nx[0], box[1] / nx[1]};
	float cour = sqrtf(1.0f / (1.0f / (dx[0] * dx[0]) + 1.0f / (dx[1] * dx[1])));
	if (dt >= cour)
	{
		fprintf(stderr, "Invalid timestep, courant condition violation, dtmax = %f \n", cour);
		exit(-1);
	}

	// Inject particles in the simulation within the process boundaries
	const int range[][2] = {{0, nx[0]}, {0, nx[1]}};
	for (int n = 0; n < n_species; ++n)
		spec_inject_particles(&species[n].main_vector, range, sim->proc_limits, species[n].ppc,
		                      &species[n].density, species[n].dx, species[n].n_move, species[n].ufl,
		                      species[n].uth);

	// Initialise the regions
	sim->n_regions = n_regions;
	sim->regions = malloc(n_regions * sizeof(t_region));
	assert(sim->regions);

	t_region *prev = NULL;
	for (int i = 0; i < n_regions; i++)
	{
		t_region *next = (i == n_regions - 1) ? NULL : &sim->regions[i + 1];
		region_new(&sim->regions[i], i, n_regions, sim->proc_nx, sim->proc_limits, sim->proc_box,
		           n_species, species, dt, sim->on_right_edge, sim->on_left_edge, prev, next);
		prev = &sim->regions[i];
	}

	// Cleaning particles species
	for (int n = 0; n < n_species; ++n)
		spec_delete(&species[n]);

	// Link each region in the process with all its neighbours
	for (int i = 0; i < n_regions; i++)
	{
		region_link_adj_part(&sim->regions[i]);
		region_link_adj_grid(&sim->regions[i]);
	}

	// Calculate the particle initial energy
	for (int i = 0; i < n_regions; i++)
		for (int n = 0; n < n_species; n++)
			spec_calculate_energy(&sim->regions[i].species[n]);

	// Create output directory
	sim_create_dir(sim);

	char filename[128];
	FILE *file;
	sprintf(filename, "output/%s/energy.csv", sim->name);
	file = fopen(filename, "w+");
	fclose(file);

	sprintf(filename, "output/%s/region_timings.csv", sim->name);
	file = fopen(filename, "w+");
	fclose(file);

#ifdef ENABLE_TASKING
	// Init the communication task management mechanism
	init_task_management();
#endif
}

void sim_delete(t_simulation *sim)
{
	for (int i = 0; i < sim->n_regions; i++)
		region_delete(&sim->regions[i]);
	free(sim->regions);

#ifdef ENABLE_TASKING
	delete_task_management();
#endif
}

void sim_add_laser(t_simulation *sim, t_emf_laser *laser)
{
	const int sim_nrow = sim->nx[0] + sim->gc[0][0] + sim->gc[0][1];
	const int sim_size = sim_nrow * (sim->nx[1] + sim->gc[1][0] + sim->gc[1][1]);

	// Add laser in the simulation space
	t_vfld *restrict E_sim = calloc(sim_size, sizeof(t_vfld));
	t_vfld *restrict B_sim = calloc(sim_size, sizeof(t_vfld));

	emf_add_laser(laser, E_sim + 1 + sim_nrow, B_sim + 1 + sim_nrow, sim->nx, sim_nrow,
	              sim->regions->local_emf.dx, sim->gc);

	// Copy the resulting EMF to each region
	// WARNING: Only support a single laser
	for (int region_id = 0; region_id < sim->n_regions; ++region_id)
	{
		t_region *region = &sim->regions[region_id];
		t_vfld *E_region = E_sim + region->limits[0][0] + region->limits[1][0] * sim_nrow;
		t_vfld *B_region = B_sim + region->limits[0][0] + region->limits[1][0] * sim_nrow;
		t_emf *emf_region = &region->local_emf;
		const int nrow = emf_region->nrow;

		for (int j = 0; j < emf_region->nx[1] + emf_region->gc[1][1] + emf_region->gc[1][0]; ++j)
		{
			memcpy(emf_region->B_buf + j * nrow, B_region, nrow * sizeof(t_vfld));
			memcpy(emf_region->E_buf + j * nrow, E_region, nrow * sizeof(t_vfld));
			B_region += sim_nrow;
			E_region += sim_nrow;
		}
	}

	free(B_sim);
	free(E_sim);
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

	for (int i = 0; i < sim->n_regions; i++)
		sim->regions[i].local_current.smooth = *smooth;
}

void sim_set_moving_window(t_simulation *sim)
{
	sim->moving_window = true;
	for (int i = 0; i < sim->n_regions; i++)
		region_set_moving_window(&sim->regions[i]);
}

/*********************************************************************************************
 Iteration
 *********************************************************************************************/
void sim_iter(t_simulation *sim)
{
	t_region *regions = sim->regions;
	const int n_regions = sim->n_regions;
	const t_smooth filter = regions->local_current.smooth;

	sim->iter++;

	for (int i = 0; i < n_regions; i++)
	{
		current_zero(&regions[i].local_current);

		for (int k = 0; k < regions[i].n_species; k++)
			spec_advance(&regions[i].species[k], &regions[i].local_emf, &regions[i].local_current,
			             regions[i].limits, sim->nx);

		if (i == 0 || i == n_regions - 1)
			current_exchange_gc_y(&regions[i].local_current, sim->adj_ranks_grid);
	}

	for (int i = 0; i < n_regions; i++)
	{
		current_reduction_y(&regions[i].local_current);

		for (int k = 0; k < regions[i].n_species; k++)
			spec_send_outgoing_np(&regions[i].species[k], i, k, sim->adj_ranks_part);
	}

	for (int i = 0; i < n_regions; i++)
	{
		current_exchange_gc_x(&regions[i].local_current, i, sim->adj_ranks_grid);

		for (int k = 0; k < regions[i].n_species; k++)
			spec_send_particles(&regions[i].species[k], i, k, sim->adj_ranks_part);
	}

	for (int i = 0; i < n_regions; i++)
		current_reduction_x(&regions[i].local_current);

	if (filter.xtype != NONE)
	{
		for (int k = 0; k < filter.xlevel; k++)
		{
			for (int i = 0; i < n_regions; i++)
			{
				current_smooth_x(&regions[i].local_current, BINOMIAL);
				current_exchange_gc_x(&regions[i].local_current, i, sim->adj_ranks_grid);
			}

			for (int i = 0; i < n_regions; i++)
				current_update_gc_x(&regions[i].local_current);
		}

		if (filter.xtype == COMPENSATED)
		{
			for (int i = 0; i < n_regions; i++)
			{
				current_smooth_x(&regions[i].local_current, COMPENSATED);
				current_exchange_gc_x(&regions[i].local_current, i, sim->adj_ranks_grid);
			}

			for (int i = 0; i < n_regions; i++)
				current_update_gc_x(&regions[i].local_current);
		}
	}

//		if (filter.ytype != NONE)
//		{
//			for (int k = 0; k < filter.ylevel; k++)
//			{
//				for(int i = 0; i < n_regions; i++)
//					current_smooth_y(&regions[i].local_current, BINOMIAL);
//
//				for(int i = 0; i < n_regions; i++)
//					current_gc_update_y(&regions[i].local_current);
//			}
//
//			if (filter.ytype == COMPENSATED)
//			{
//				for(int i = 0; i < n_regions; i++)
//					current_smooth_y(&regions[i].local_current, COMPENSATED);
//
//				for(int i = 0; i < n_regions; i++)
//					current_gc_update_y(&regions[i].local_current);
//			}
//		}

	for (int i = 0; i < n_regions; i++)
	{
		emf_advance(&regions[i].local_emf, &regions[i].local_current);
		emf_exchange_gc_x(&regions[i].local_emf, i, sim->adj_ranks_grid);
	}

	for (int i = 0; i < n_regions; i++)
	{
		emf_update_gc_x(&regions[i].local_emf);

		if (i == 0 || i == n_regions - 1)
			emf_exchange_gc_y(&regions[i].local_emf, sim->adj_ranks_grid);
	}

	for (int i = 0; i < n_regions; i++)
	{
		emf_update_gc_y(&regions[i].local_emf);

		for (int k = 0; k < regions[i].n_species; k++)
			spec_receive_particles(&regions[i].species[k]);
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

void sim_timings(t_simulation *sim, uint64_t t0, uint64_t t1)
{
	int npart = 0;

#ifdef ENABLE_TASKING
	const int num_threads = nanos6_get_num_cpus();
#else
	const int num_threads = 1;
#endif

	for (int j = 0; j < sim->n_regions; j++)
		for (int i = 0; i < sim->regions[j].n_species; i++)
			npart += sim->regions[j].species[i].main_vector.size;

#ifndef TEST
	fprintf(stdout, "Topology: %d x %d\n", sim->num_procs_cart[0], sim->num_procs_cart[1]);
	fprintf(stdout, "Simulation: %s\n", sim->name);
	fprintf(stdout, "Number of regions: %d\n", sim->n_regions);
	fprintf(stdout, "Number of processes: %d\n", sim->num_procs);
	fprintf(stdout, "Number of threads per process: %d\n", num_threads);
//	fprintf(stdout, "Time for spec. advance = %f s\n", spec_time() / n_threads); // Disable due to compatibility issues
//	fprintf(stdout, "Time for emf   advance = %f s\n", emf_time() / n_threads); // Disable due to compatibility issues
	fprintf(stdout, "Total simulation time  = %f s\n", timer_interval_seconds(t0, t1));
	fprintf(stdout, "\n");

	// Disable due to compatibility issues
//	if (spec_time() > 0)
//	{
//		double perf = spec_perf();
//		fprintf(stderr, "Particle advance [nsec/part] = %f \n", 1.e9 * perf);
//		fprintf(stderr, "Particle advance [Mpart/sec] = %f \n", 1.e-6 / perf);
//	}

#else
	printf("%s,%d,%d,%d,%f\n", sim->name, sim->num_procs, num_threads, sim->n_regions, timer_interval_seconds(t0, t1));
#endif

}

// Save the simulation energy to a CSV file
void sim_report_energy(t_simulation *sim)
{
	char filename[128];

	int i;
	double tot_emf = 0;
	double tot_part = 0;

	for (int j = 0; j < sim->n_regions; j++)
	{
		tot_emf += emf_get_energy(&sim->regions[j].local_emf);

		for (i = 0; i < sim->regions[j].n_species; i++)
		{
			spec_calculate_energy(&sim->regions[j].species[i]);
			tot_part += sim->regions[j].species[i].energy;
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

// Save the grid quantity to a ZDF file
void sim_report_grid_zdf(t_simulation *sim, enum report_grid_type type, const int coord)
{
	char path[128] = "";
	sprintf(path, "output/%s/grid", sim->name);
	const int buf_size = sim->nx[0] * sim->nx[1];
	t_fld *restrict buf = calloc(buf_size, sizeof(t_fld));

	switch (type)
	{
		case REPORT_BFLD:
			for (int j = 0; j < sim->n_regions; j++)
			{
				int offset_y = sim->regions[j].limits[1][0];
				int offset_x = sim->regions[j].limits[0][0];
				emf_reconstruct_global_buffer(&sim->regions[j].local_emf, buf,
				                            offset_y, offset_x, sim->nx[0], BFLD, coord);
			}


			if(sim->proc_rank == ROOT)
				MPI_Reduce(MPI_IN_PLACE, buf, buf_size, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);
			else MPI_Reduce(buf, NULL, buf_size, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);

			if (sim->proc_rank == ROOT)
				emf_report(buf, sim->box, sim->nx, sim->iter, sim->dt, BFLD, coord, path);
			break;

		case REPORT_EFLD:
			for (int j = 0; j < sim->n_regions; j++)
			{
				int offset_y = sim->regions[j].limits[1][0];
				int offset_x = sim->regions[j].limits[0][0];
				emf_reconstruct_global_buffer(&sim->regions[j].local_emf, buf,
				                            offset_y, offset_x, sim->nx[0], EFLD, coord);
			}

			if(sim->proc_rank == ROOT)
				MPI_Reduce(MPI_IN_PLACE, buf, buf_size, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);
			else MPI_Reduce(buf, NULL, buf_size, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);

			if (sim->proc_rank == ROOT)
				emf_report(buf, sim->box, sim->nx, sim->iter, sim->dt, EFLD, coord, path);
			break;

		case REPORT_CURRENT:
			for (int j = 0; j < sim->n_regions; j++)
			{
				int offset_y = sim->regions[j].limits[1][0];
				int offset_x = sim->regions[j].limits[0][0];
				current_reconstruct_global_buffer(&sim->regions[j].local_current, buf,
				                                  offset_y, offset_x, sim->nx[0], coord);
			}

			if(sim->proc_rank == ROOT)
				MPI_Reduce(MPI_IN_PLACE, buf, buf_size, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);
			else MPI_Reduce(buf, NULL, buf_size, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);

			if (sim->proc_rank == ROOT)
				current_report(buf, sim->iter, sim->nx, sim->box, sim->dt, coord, path);
			break;

		default:
			fprintf(stderr, "Error: Unsupported grid report!");
			break;
	}

	free(buf);
}

// Save a particle property to a ZDF file
void sim_report_spec_zdf(t_simulation *sim, const int species, const int rep_type,
                         const int pha_nx[2], const float pha_range[][2])
{
	char path[128] = "";
	sprintf(path, "output/%s/%s", sim->name, sim->regions->species[species].name);

	switch (rep_type & 0xF000)
	{
		case CHARGE:
		{
			size_t buf_size = (sim->nx[0] + 1) * (sim->nx[1] + 1);
			t_part_data *charge = calloc(buf_size, sizeof(t_part_data));

			for (int j = 0; j < sim->n_regions; j++)
				spec_deposit_charge(&sim->regions[j].species[species], charge, sim->nx[0] + 1);

			if(sim->proc_rank == ROOT)
				MPI_Reduce(MPI_IN_PLACE, charge, buf_size, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);
			else MPI_Reduce(charge, NULL, buf_size, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);


			if (sim->proc_rank == ROOT)
				spec_rep_charge(charge, sim->nx, sim->box, sim->iter, sim->dt, sim->moving_window, path);
			free(charge);
		}
		break;

		case PHA:
		{
			float *buf = calloc(pha_nx[0] * pha_nx[1], sizeof(float));

			for(int j = 0; j < sim->n_regions; j++)
				spec_deposit_pha(&sim->regions[j].species[species], rep_type, pha_nx, pha_range, buf);

			if(sim->proc_rank == ROOT)
				MPI_Reduce(MPI_IN_PLACE, buf, pha_nx[0] * pha_nx[1], MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);
			else MPI_Reduce(buf, NULL, pha_nx[0] * pha_nx[1], MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);

			if (sim->proc_rank == ROOT)
				spec_rep_pha(buf, rep_type, pha_nx, pha_range, sim->iter, sim->dt, path);
			free(buf);
		}
		break;

//		case PARTICLES:
//			const char *quants[] = {"x1", "x2", "u1", "u2", "u3"};
//			const char *units[] = {"c/\\omega_p", "c/\\omega_p", "c", "c", "c"};
//
//			t_zdf_iteration iter = {.n = sim->iter, .t = sim->iter * sim->dt, .time_units = "1/\\omega_p"};
//
//			// Allocate buffer for positions
//			int np = 0;
//			for(int j = 0; j < sim->n_regions; j++)
//				np += sim->regions[j].species[species].main_vector.size;
//
//			size_t size = np * sizeof(float);
//			float *data = malloc(size);
//
//			t_zdf_part_info info = {.name = (char*) sim->name, .nquants = 5, .quants = (char**) quants,
//									.units = (char**) units, .np = np};
//
//			// Create file and add description
//			t_zdf_file part_file;
//			zdf_part_file_open(&part_file, &info, &iter, path);
//
//			// Add positions and generalized velocities
//			t_species *restrict spec;
//			int offset;
//
//			// x1
//			offset = 0;
//			for(int j = 0; j < sim->n_regions; j++)
//			{
//				spec = &sim->regions[j].species[species];
//				for (int i = 0; i < spec->main_vector.size; i++)
//					data[i + offset] = (spec->n_move + spec->main_vector.data[i].ix + spec->main_vector.data[i].x)
//							* spec->dx[0];
//				offset += spec->main_vector.size;
//			}
//
//			zdf_part_file_add_quant(&part_file, quants[0], data, np);
//
//			// x2
//			offset = 0;
//			for(int j = 0; j < sim->n_regions; j++)
//			{
//				spec = &sim->regions[j].species[species];
//				for (int i = 0; i < spec->main_vector.size; i++)
//					data[i + offset] = (spec->main_vector.data[i].iy + spec->main_vector.data[i].y) * spec->dx[1];
//				offset += spec->main_vector.size;
//			}
//
//			zdf_part_file_add_quant(&part_file, quants[1], data, np);
//
//			// ux
//			offset = 0;
//			for(int j = 0; j < sim->n_regions; j++)
//			{
//				spec = &sim->regions[j].species[species];
//				for (int i = 0; i < spec->main_vector.size; i++)
//					data[i + offset] = spec->main_vector.data[i].ux;
//				offset += spec->main_vector.size;
//			}
//
//			zdf_part_file_add_quant(&part_file, quants[2], data, np);
//
//			// uy
//			offset = 0;
//			for(int j = 0; j < sim->n_regions; j++)
//			{
//				spec = &sim->regions[j].species[species];
//				for (int i = 0; i < spec->main_vector.size; i++)
//					data[i + offset] = spec->main_vector.data[i].uy;
//				offset += spec->main_vector.size;
//			}
//
//			zdf_part_file_add_quant(&part_file, quants[3], data, np);
//
//			// uz
//			offset = 0;
//			for(int j = 0; j < sim->n_regions; j++)
//			{
//				spec = &sim->regions[j].species[species];
//				for (int i = 0; i < spec->main_vector.size; i++)
//					data[i + offset] = spec->main_vector.data[i].uz;
//				offset += spec->main_vector.size;
//			}
//
//			zdf_part_file_add_quant(&part_file, quants[4], data, np);
//
//			free(data);
//			zdf_close_file(&part_file);
//			break;

		default:
			fprintf(stderr, "Error: Unsupported particle report!");
			break;
	}
}
