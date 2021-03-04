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
#include <omp.h>

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

	// Inject particles in the simulation that will be distributed to all the regions
	const int range[][2] = { { 0, nx[0] }, { 0, nx[1] } };
	for (int n = 0; n < n_species; ++n)
		spec_inject_particles(&species[n], range);

	// Initialise the regions
	sim->n_regions = n_regions;
	sim->regions = malloc(n_regions * sizeof(t_region));

	for(int i = 0; i < n_regions; i++)
	{
		t_region *prev = i == 0 ? &sim->regions[n_regions - 1] : &sim->regions[i - 1];
		t_region *next = i == n_regions - 1 ? &sim->regions[0] : &sim->regions[i + 1];
		region_new(&sim->regions[i], n_regions, nx, i, n_species, species, box, dt, prev, next);
	}

	for(int i = 0; i < n_regions; i++)
		region_init(&sim->regions[i]);

	// Cleaning
	for (int n = 0; n < n_species; ++n)
		spec_delete(&species[n]);

	// Check time step
	float dx[] = { box[0] / nx[0], box[1] / nx[1] };
	float cour = sqrtf(1.0f / (1.0f / (dx[0] * dx[0]) + 1.0f / (dx[1] * dx[1])));
	if (dt >= cour)
	{
		fprintf(stderr, "Invalid timestep, courant condition violation, dtmax = %f \n", cour);
		exit(-1);
	}

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
	for(int i = 0; i < sim->n_regions; i++)
		region_delete(&sim->regions[i]);
	free(sim->regions);
}

void sim_add_laser(t_simulation *sim, t_emf_laser *laser)
{
	region_add_laser(&sim->regions[0], laser);
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
	region_advance(sim->regions, sim->n_regions);
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

// Save the particles charge map to a CSV file
void sim_report_charge_csv(t_simulation *sim)
{
	int n_species = sim->regions[0].n_species;
	t_part_data *charge, *buf, *b, *c;
	size_t size = (sim->nx[0] + 1) * (sim->nx[1] + 1) * sizeof(t_part_data);  // Add 1 guard cell to the upper boundary
	size_t buf_size = sim->nx[0] * sim->nx[1] * sizeof(t_part_data);
	charge = malloc(size);
	buf = malloc(buf_size);

	for (int n = 0; n < n_species; n++)
	{
		memset(charge, 0, size);
		memset(buf, 0, buf_size);

		for(int i = 0; i < sim->n_regions; i++)
			spec_deposit_charge(&sim->regions[i].species[n], charge);

		// Correct boundary values
		// x
		if (!sim->moving_window) for (int j = 0; j < sim->nx[1] + 1; j++)
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
		sprintf(filename, "%s_charge_map_%d.csv", sim->regions[0].species[n].name, sim->iter);
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

	t_region *restrict region = &sim->regions[0];
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

void sim_region_timings(t_simulation *sim)
{
	char filename[128];

	sprintf(filename, "output/%s/region_timings.csv", sim->name);
	FILE *file = fopen(filename, "a+");

	if (file)
	{
		t_region *restrict region = &sim->regions[0];

		while (region->next->id != 0)
		{
			fprintf(file, "%lf;", region->iter_time);
			region = region->next;
		}

		fprintf(file, "%lf\n", region->iter_time);
		fclose(file);

	} else
	{
		printf("Error on open file: %s", filename);
		exit(1);
	}
}

void sim_timings(t_simulation *sim, uint64_t t0, uint64_t t1)
{
	int npart = 0;
	int n_regions = sim->n_regions;
	int n_threads = omp_get_max_threads();

	t_region *restrict region = &sim->regions[0];
	do
	{
		for (int i = 0; i < sim->regions[0].n_species; i++)
			npart += region->species[i].main_vector.size;
		region = region->next;
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

			spec_rep_charge(charge, sim->nx, sim->box, sim->iter, sim->dt, sim->moving_window, path);

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

			t_zdf_iteration iter = {.n = sim->iter, .t = sim->iter * sim->dt, .time_units = "1/\\omega_p"};

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

			t_zdf_part_info info = {.name = (char*) sim->name, .nquants = 5, .quants = (char**) quants,
									.units = (char**) units, .np = np};

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
					data[i + offset] = (spec->n_move + spec->main_vector.data[i].ix + spec->main_vector.data[i].x)
							* spec->dx[0];
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
					data[i + offset] = (spec->main_vector.data[i].iy + spec->main_vector.data[i].y) * spec->dx[1];
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
					data[i + offset] = spec->main_vector.data[i].ux;
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
					data[i + offset] = spec->main_vector.data[i].uy;
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
