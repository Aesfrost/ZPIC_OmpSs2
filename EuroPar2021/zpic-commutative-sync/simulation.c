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
#include <nanos6.h>

#include "simulation.h"
#include "timer.h"
#include "zdf.h"


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
	#pragma acc set device_num(0) // Dummy operation to work with the PGI Compiler

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
	float dx[] = {box[0] / nx[0], box[1] / nx[1]};
	float cour = sqrtf(1.0f / (1.0f / (dx[0] * dx[0]) + 1.0f / (dx[1] * dx[1])));
	if (dt >= cour)
	{
		fprintf(stderr, "Invalid timestep, courant condition violation, dtmax = %f \n", cour);
		exit(-1);
	}

	// Inject particles in the simulation that will be distributed to all the regions
	const int range[][2] = {{0, nx[0]}, {0, nx[1]}};
	for (int n = 0; n < n_species; ++n)
		spec_inject_particles(&species[n].main_vector, range, species[n].ppc, &species[n].density,
				species[n].dx, species[n].n_move, species[n].ufl, species[n].uth);

	//Init global current
	current_new(&sim->global_current, nx, box, dt);

	// Initialise the regions (recursively)
	sim->first_region = malloc(sizeof(t_region));
	assert(sim->first_region);
	region_new(sim->first_region, n_regions, nx, 0, n_species, species, box, dt, &sim->global_current, NULL);

	// Cleaning
	for (int n = 0; n < n_species; ++n)
		spec_delete(&species[n]);

	// Link adjacent regions
	t_region *restrict region = sim->first_region;
	do
	{
		region_link_adj_regions(region);
		region = region->next;
	} while (region->id != 0);

	// Calculate the particle initial energy
	region = sim->first_region;
	do
	{
		for (int n = 0; n < n_species; n++)
			spec_calculate_energy(&region->species[n]);
		region = region->next;
	} while (region->id != 0);

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

}

void sim_delete(t_simulation *sim)
{
	t_region *restrict region = sim->first_region->prev;
	while (region->id != 0)
	{
		region_delete(region);
		region = region->prev;
		free(region->next);
	}
	region_delete(sim->first_region);
	free(sim->first_region);

	current_delete(&sim->global_current);
}

void sim_add_laser(t_simulation *sim, t_emf_laser *laser)
{
	region_add_laser(sim->first_region, laser);
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

	t_region *restrict region = sim->first_region;
	do
	{
		region->local_current.smooth = *smooth;
		region = region->next;
	} while (region->id != 0);
}

void sim_set_moving_window(t_simulation *sim)
{
	sim->moving_window = true;

	t_region *restrict region = sim->first_region;
	do
	{
		region_set_moving_window(region);
		region = region->next;
	} while (region->id != 0);
}

/*********************************************************************************************
 Iteration
 *********************************************************************************************/
void sim_iter(t_simulation *sim)
{
	current_zero(&sim->global_current);

	// Advance one iteration in each region (recursively)
	region_advance(sim->first_region);
	sim->iter++;

	#pragma oss taskwait
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
	int n_regions = 0;
	int n_threads = nanos6_get_num_cpus();

	t_region *restrict region = sim->first_region;
	do
	{
		for (int i = 0; i < sim->first_region->n_species; i++)
		npart += region->species[i].main_vector.size;
		region = region->next;
		n_regions++;
	}while (region->id != 0);

#ifndef TEST
	fprintf(stdout, "Simulation: %s\n", sim->name);
	fprintf(stdout, "Number of regions: %d\n", n_regions);
	fprintf(stdout, "Number of threads: %d\n", n_threads);
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
	printf("%s,%d,%d,%f\n", sim->name, n_regions, n_threads, timer_interval_seconds(t0, t1));
#endif
}

void save_data_csv(t_fld *grid, unsigned int sizeX, unsigned int sizeY, const char filename[128],
		const char sim_name[64])
{
	static bool dir_exists = false;

	char fullpath[256];

	if(!dir_exists)
	{
		//Create the output directory if it doesn't exists
		strcpy(fullpath, "output");
		struct stat sb;
		if (stat(fullpath, &sb) == -1)
		{
			mkdir(fullpath, 0700);
		}

		strcat(fullpath, "/");
		strcat(fullpath, sim_name);
		if (stat(fullpath, &sb) == -1)
		{
			mkdir(fullpath, 0700);
		}
	}else
	{
		strcpy(fullpath, "output/");
		strcat(fullpath, sim_name);
	}

	strcat(fullpath, "/");
	strcat(fullpath, filename);

	FILE *file = fopen(fullpath, "wb+");

	if (file != NULL)
	{
		for (unsigned int j = 0; j < sizeY; j++)
		{
			for (unsigned int i = 0; i < sizeX - 1; i++)
			{
				fprintf(file, "%f;", grid[i + j * sizeX]);
			}
			fprintf(file, "%f\n", grid[(j + 1) * sizeX - 1]);
		}
	} else
	{
		printf("Couldn't open %s", filename);
		exit(1);
	}

	fclose(file);
}

// Save the simulation energy to a CSV file
void sim_report_energy(t_simulation *sim)
{
	char filename[128];

	int i;
	double tot_emf = 0;
	double tot_part = 0;

	t_region *restrict region = sim->first_region;
	do
	{
		tot_emf += emf_get_energy(&region->local_emf);

		for (i = 0; i < sim->first_region->n_species; i++)
			tot_part += region->species[i].energy;
		region = region->next;
	} while (region->id != 0);

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

// Save the particles charge map to a CSV file
void sim_report_charge_csv(t_simulation *sim)
{
	t_region *restrict region;
	int n_species = sim->first_region->n_species;
	t_part_data *charge, *buf, *b, *c;
	size_t size = (sim->nx[0] + 1) * (sim->nx[1] + 1) * sizeof(t_part_data);   // Add 1 guard cell to the upper boundary
	size_t buf_size = sim->nx[0] * sim->nx[1] * sizeof(t_part_data);
	charge = malloc(size);
	buf = malloc(buf_size);

	for (int n = 0; n < n_species; n++)
	{
		memset(charge, 0, size);
		memset(buf, 0, buf_size);

		region = sim->first_region;

		do
		{
			spec_deposit_charge(&region->species[n], charge);
			region = region->next;
		} while (region->id != 0);

		// Correct boundary values
		// x
		if (!sim->moving_window)
			for (int j = 0; j < sim->nx[1] + 1; j++)
				charge[0 + j * (sim->nx[0] + 1)] += charge[sim->nx[0] + j * (sim->nx[0] + 1)];

		// y - Periodic boundaries
		for (int i = 0; i < sim->nx[0] + 1; i++)
			charge[i] += charge[i + sim->nx[1] * (sim->nx[0] + 1)];

		b = buf;
		c = charge;

		for (int j = 0; j < sim->nx[1]; j++)
		{
			for (int i = 0; i < sim->nx[0]; i++)
				b[i] = c[i];

			b += sim->nx[0];
			c += sim->nx[0] + 1;
		}

		char filename[128];
		sprintf(filename, "%s_charge_map_%d.csv", sim->first_region->species[n].name, sim->iter);
		save_data_csv(buf, sim->nx[0], sim->nx[1], filename, sim->name);
	}

	free(charge);
	free(buf);
}

// Save the EMF magnitude to a CSV version
void sim_report_emf_csv(t_simulation *sim)
{
	char filenameE[128];
	char filenameB[128];

	t_fld *restrict E_magnitude = malloc(sim->nx[0] * sim->nx[1] * sizeof(t_fld));
	t_fld *restrict B_magnitude = malloc(sim->nx[0] * sim->nx[1] * sizeof(t_fld));

	t_region *restrict region = sim->first_region;
	do
	{
		emf_report_magnitude(&region->local_emf, E_magnitude, B_magnitude, sim->nx[0], region->limits_y[0]);
		region = region->next;
	} while (region->id != 0);

	sprintf(filenameE, "e_mag_map_%d.csv", sim->iter);
	sprintf(filenameB, "b_mag_map_%d.csv", sim->iter);

	save_data_csv(E_magnitude, sim->nx[0], sim->nx[1], filenameE, sim->name);
	save_data_csv(B_magnitude, sim->nx[0], sim->nx[1], filenameB, sim->name);

	free(E_magnitude);
	free(B_magnitude);
}

// Save the region time in a CSV file (Disabled due to a compatibility issue)
//void sim_region_timings(t_simulation *sim)
//{
//	char filename[128];
//
//	sprintf(filename, "output/%s/region_timings.csv", sim->name);
//	FILE *file = fopen(filename, "a+");
//
//	if (file)
//	{
//		t_region *restrict region = sim->first_region;
//
//		while(region->next->id != 0)
//		{
//			fprintf(file, "%lf;", region->iter_time);
//			region = region->next;
//		}
//
//		fprintf(file, "%lf\n", region->iter_time);
//		fclose(file);
//
//	} else
//	{
//		printf("Error on open file: %s", filename);
//		exit(1);
//	}
//}

// Save the grid quantity to a ZDF file
void sim_report_grid_zdf(t_simulation *sim, enum report_grid_type type, const int coord)
{
	t_region *restrict region = sim->first_region;
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
				current_reconstruct_global_buffer(&region->local_current, global_buf, region->limits_y[0], coord);
				region = region->next;
			} while (region->id != 0);

			current_report(global_buf, sim->iter, sim->nx, sim->box, sim->dt, coord, path);
			break;

		default:
			break;
	}

	free(global_buf);
}

// Wrapper for CSV report
void sim_report_csv(t_simulation *sim)
{
	sim_report_emf_csv(sim);
	sim_report_charge_csv(sim);
}

// Save a particle property to a ZDF file
void sim_report_spec_zdf(t_simulation *sim, const int species, const int rep_type,
		const int pha_nx[2], const float pha_range[][2])
{
	size_t size;
	t_region *restrict region = sim->first_region;
	char path[128] = "";
	sprintf(path, "output/%s/%s", sim->name, sim->first_region->species[species].name);

	switch (rep_type & 0xF000)
	{
		case CHARGE:
		{
			size = (sim->nx[0] + 1) * (sim->nx[1] + 1) * sizeof(t_part_data);  // Add 1 guard cell to the upper boundary
			t_part_data *restrict charge = malloc(size);
			memset(charge, 0, size);

			region = sim->first_region;
			do
			{
				spec_deposit_charge(&region->species[species], charge);
				region = region->next;
			} while (region->id != 0);

			spec_rep_charge(charge, sim->nx, sim->box, sim->iter, sim->dt, sim->moving_window, path);

			free(charge);
		}
			break;

		case PHA:
		{
			float *buf = malloc(pha_nx[0] * pha_nx[1] * sizeof(float));
			memset(buf, 0, pha_nx[0] * pha_nx[1] * sizeof(float));

			region = sim->first_region;
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

			t_zdf_iteration iter = {.n = sim->iter, .t = sim->iter * sim->dt, .time_units = "1/\\omega_p"};

			// Allocate buffer for positions
			int np = 0;
			region = sim->first_region;
			do
			{
				np += region->species[species].main_vector.size;
				region = region->next;
			} while (region->id != 0);

			size = np * sizeof(float);
			float *data = malloc(size);

			t_zdf_part_info info = {.name = (char*) sim->name, .nquants = 5, .quants = (char**) quants,
									.units = (char**) units, .np = np};

			// Create file and add description
			t_zdf_file part_file;
			zdf_part_file_open(&part_file, &info, &iter, path);

			// Add positions and generalized velocities
			t_species *restrict spec;
			int offset;

			// x1
			region = sim->first_region;
			offset = 0;
			do
			{
				spec = &region->species[species];
				for (int i = 0; i < spec->main_vector.size; i++)
					data[i + offset] = (spec->n_move + spec->main_vector.data[i].ix + spec->main_vector.data[i].x)
							* spec->dx[0];
				region = region->next;
				offset += spec->main_vector.size;
			} while (region->id != 0);
			zdf_part_file_add_quant(&part_file, quants[0], data, np);

			// x2
			region = sim->first_region;
			offset = 0;
			do
			{
				spec = &region->species[species];
				for (int i = 0; i < spec->main_vector.size; i++)
					data[i + offset] = (spec->main_vector.data[i].iy + spec->main_vector.data[i].y) * spec->dx[1];
				region = region->next;
				offset += spec->main_vector.size;
			} while (region->id != 0);
			zdf_part_file_add_quant(&part_file, quants[1], data, np);

			// ux
			region = sim->first_region;
			offset = 0;
			do
			{
				spec = &region->species[species];
				for (int i = 0; i < spec->main_vector.size; i++)
					data[i + offset] = spec->main_vector.data[i].ux;
				region = region->next;
				offset += spec->main_vector.size;
			} while (region->id != 0);
			zdf_part_file_add_quant(&part_file, quants[2], data, np);

			// uy
			region = sim->first_region;
			offset = 0;
			do
			{
				spec = &region->species[species];
				for (int i = 0; i < spec->main_vector.size; i++)
					data[i + offset] = spec->main_vector.data[i].uy;
				region = region->next;
				offset += spec->main_vector.size;
			} while (region->id != 0);
			zdf_part_file_add_quant(&part_file, quants[3], data, np);

			// uz
			region = sim->first_region;
			offset = 0;
			do
			{
				spec = &region->species[species];
				for (int i = 0; i < spec->main_vector.size; i++)
					data[i + offset] = spec->main_vector.data[i].uz;
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
