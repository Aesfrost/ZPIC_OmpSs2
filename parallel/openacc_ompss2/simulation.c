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
#include "csv_handler.h"

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

void sim_iter(t_simulation *sim)
{
	// Advance one iteration in each region (recursively)
	region_advance(sim->first_region);
	sim->iter++;
}

void sim_timings(t_simulation *sim, uint64_t t0, uint64_t t1)
{
	int npart = 0;
	int n_regions = 0;
	int gpu_regions = 0;
	int n_threads = 12; //nanos6_get_num_cpus();

	t_region *restrict region = sim->first_region;
	do
	{
		for (int i = 0; i < sim->first_region->n_species; i++)
		npart += region->species[i].main_vector.size;
		region = region->next;
		n_regions++;
		if(region->enable_gpu) gpu_regions++;
	}while (region->id != 0);

	fprintf(stdout, "Simulation: %s\n", sim->name);
	fprintf(stdout, "Number of regions (Total): %d\n", get_n_regions());
	fprintf(stdout, "Number of regions (GPU): %d (effective: %d regions)\n", gpu_regions, get_gpu_regions_effective());
	fprintf(stdout, "Number of threads: %d\n", n_threads);
	fprintf(stdout, "Time for spec. advance = %f s\n", spec_time() / n_threads);
	fprintf(stdout, "Time for emf   advance = %f s\n", emf_time() / n_threads);
	fprintf(stdout, "Total simulation time  = %f s\n", timer_interval_seconds(t0, t1));
	fprintf(stdout, "\n");

	if (spec_time() > 0)
	{
		double perf = spec_perf();
		fprintf(stderr, "Particle advance [nsec/part] = %f \n", 1.e9 * perf);
		fprintf(stderr, "Particle advance [Mpart/sec] = %f \n", 1.e-6 / perf);
	}
}

void sim_new(t_simulation *sim, int nx[], float box[], float dt, float tmax, int ndump,
		t_species *species, int n_species, char name[64], int n_regions, float gpu_percentage, int n_gpu_regions)
{
	t_particle_vector *restrict part;
	double usq, gamma;

	sim->iter = 0;
	sim->moving_window = false;
	sim->dt = dt;
	sim->tmax = tmax;
	sim->ndump = ndump;
	sim->nx[0] = nx[0];
	sim->nx[1] = nx[1];
	strncpy(sim->name, name, 64);

	// Inject particles in the simulation that will be distributed to all the regions
	const int range[][2] = {{0, nx[0]}, {0, nx[1]}};
	for (int n = 0; n < n_species; ++n)
		spec_inject_particles(&species[n], range);

	// Initialise the regions (recursively)
	sim->first_region = malloc(sizeof(t_region));
	assert(sim->first_region);
	region_new(sim->first_region, n_regions, nx, 0, n_species, species, box, dt, gpu_percentage, n_gpu_regions, NULL);

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
//	region = sim->first_region;
//	do
//	{
//		for (int n = 0; n < n_species; n++)
//		{
//			part = &region->species[n].main_vector;
//			region->species[n].energy = 0;
//
//			for (int i = 0; i < part->size; i++)
//			{
//				usq = part->part[i].ux * part->part[i].ux + part->part[i].uy * part->part[i].uy
//						+ part->part[i].uz * part->part[i].uz;
//				gamma = sqrtf(1 + usq);
//				region->species[n].energy += usq / (gamma + 1);
//			}
//		}
//
//		region = region->next;
//	} while (region->id != 0);

	// Check time step
	float dx[] = {box[0] / nx[0], box[1] / nx[1]};
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
	t_region *restrict region = sim->first_region->prev;
	while (region->id != 0)
	{
		region_delete(region);
		region = region->prev;
		free(region->next);
	}
	region_delete(sim->first_region);
	free(sim->first_region);
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

void sim_report_charge(t_simulation *sim)
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
			region_charge_report(region, charge, n);
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

void sim_report_emf(t_simulation *sim)
{
	char filenameE[128];
	char filenameB[128];

	t_fld *restrict E_magnitude = malloc(sim->nx[0] * sim->nx[1] * sizeof(t_fld));
	t_fld *restrict B_magnitude = malloc(sim->nx[0] * sim->nx[1] * sizeof(t_fld));

	t_region *restrict region = sim->first_region;
	do
	{
		region_emf_report(region, E_magnitude, B_magnitude, sim->nx[0]);
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
		t_region *restrict region = sim->first_region;

		while(region->next->id != 0)
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
