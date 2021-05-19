/*********************************************************************************************
 ZPIC
 simulation.c

 Created by Ricardo Fonseca on 12/8/10.
 Modified by Nicolas Guidotti on 11/06/20

 Copyright 2010 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "simulation.h"
#include "timer.h"

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
	// Advance particles and deposit current
	current_zero(&sim->current);
	for (int i = 0; i < sim->n_species; i++)
		spec_advance(&sim->species[i], &sim->emf, &sim->current);

	// Update current boundary conditions and advance iteration
	current_update(&sim->current);

	// Advance EM fields
	emf_advance(&sim->emf, &sim->current);
}

void sim_new(t_simulation *sim, int nx[], float box[], float dt, float tmax, int ndump, t_species *species,
		int n_species, char name[64])
{
	t_species *restrict part;
	double usq, gamma;

	sim->dt = dt;
	sim->tmax = tmax;
	sim->ndump = ndump;
	strncpy(sim->name, name, 64);

	emf_new(&sim->emf, nx, box, dt);
	current_new(&sim->current, nx, box, dt);

	sim->n_species = n_species;
	sim->species = species;

	for (int n = 0; n < n_species; n++)
		spec_calculate_energy(&sim->species[n]);

	// Check time step
	float cour = sqrtf(1.0f / (1.0f / (sim->emf.dx[0] * sim->emf.dx[0]) + 1.0f / (sim->emf.dx[1] * sim->emf.dx[1])));
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
}

void sim_add_laser(t_simulation *sim, t_emf_laser *laser)
{
	emf_add_laser(&sim->emf, laser);
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

	sim->current.smooth = *smooth;
}

void sim_set_moving_window(t_simulation *sim)
{

	sim->emf.moving_window = 1;
	sim->current.moving_window = 1;

	int i;
	for (i = 0; i < sim->n_species; i++)
		sim->species[i].moving_window = 1;

}

void sim_delete(t_simulation *sim)
{

	int i;
	for (i = 0; i < sim->n_species; i++)
		spec_delete(&sim->species[i]);

	free(sim->species);

	current_delete(&sim->current);
	emf_delete(&sim->emf);

}

void sim_report_energy(t_simulation *sim)
{
	char filename[128];
	FILE *file;

	int i;
	double part_energy[sim->n_species];

	double tot_emf = emf_get_energy(&sim->emf);
	double tot_part = 0;

	for (i = 0; i < sim->n_species; i++)
	{
		part_energy[i] = sim->species[i].energy;
		tot_part += part_energy[i];
	}

	sprintf(filename, "output/%s/energy.csv", sim->name);
	file = fopen(filename, "a+");

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

void sim_timings(t_simulation *sim, uint64_t t0, uint64_t t1)
{
	int npart = 0;
	int i;

	for (i = 0; i < sim->n_species; i++)
		npart += sim->species[i].np;

	fprintf(stdout, "Time for spec. advance = %f s\n", spec_time());
	fprintf(stdout, "Time for emf   advance = %f s\n", emf_time());
	fprintf(stdout, "Total simulation time  = %f s\n", timer_interval_seconds(t0, t1));
	fprintf(stdout, "\n");

	if (spec_time() > 0)
	{
		double perf = spec_perf();
		fprintf(stderr, "Particle advance [nsec/part] = %f \n", 1.e9 * perf);
		fprintf(stderr, "Particle advance [Mpart/sec] = %f \n", 1.e-6 / perf);
	}
}

void sim_report_grid_zdf(t_simulation *sim, enum report_grid_type type, const int coord)
{
	char path[128];
	sprintf(path, "output/%s/grid", sim->name);

	switch (type) {
		case REPORT_BFLD:
			emf_report(&sim->emf, BFLD, coord, path);
			break;
		case REPORT_EFLD:
			emf_report(&sim->emf, EFLD, coord, path);
			break;
		case REPORT_CURRENT:
			current_report(&sim->current, coord, path);
			break;
		default:
			break;
	}
}

void sim_report_spec_zdf(t_simulation *sim, const int species, const int rep_type, const int pha_nx[],
		const float pha_range[][2])
{
	char path[128];
	sprintf(path, "output/%s/%s", sim->name, sim->species[species].name);
	spec_report(&sim->species[species], rep_type, pha_nx, pha_range, path);
}

void sim_report_csv(t_simulation *sim)
{
	emf_report_magnitude(&sim->emf, sim->name);

	for(int i = 0; i < sim->n_species; i++)
		spec_report_csv(&sim->species[i], sim->name);
}


