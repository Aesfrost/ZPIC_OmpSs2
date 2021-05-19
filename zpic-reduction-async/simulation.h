/*********************************************************************************************
 ZPIC
 simulation.h

 Created by Ricardo Fonseca on 11/8/10.
 Modified by Nicolas Guidotti on 11/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#ifndef __SIMULATION__
#define __SIMULATION__

#include <stdint.h>

#include "region.h"
#include "particles.h"
#include "emf.h"
#include "current.h"

enum report_grid_type {
	REPORT_EFLD, REPORT_BFLD, REPORT_CURRENT
};

typedef struct {
	char name[64];

	// Time step
	float dt;
	float tmax;

	// Diagnostic frequency
	int ndump;

	// Simulation data
	int nx[2];
	t_fld box[2];
	bool moving_window;
	t_region *first_region; // Pointer to the first region (id = 0) in a double-linked list

	int iter;

} t_simulation;

// Setup
void sim_new(t_simulation *sim, int nx[2], float box[2], float dt, float tmax, int ndump, t_species *species,
		int n_species, char name[64], int n_regions);
void sim_init(t_simulation *sim, int n_regions);
void sim_set_moving_window(t_simulation *sim);
void sim_set_smooth(t_simulation *sim, t_smooth *smooth);
void sim_add_laser(t_simulation *sim, t_emf_laser *laser);
void sim_delete(t_simulation *sim);

// Iteration
void sim_iter(t_simulation *sim);

// Report
int report(int n, int ndump);
void sim_report(t_simulation *sim);
void sim_report_energy(t_simulation *sim);
void sim_timings(t_simulation *sim, uint64_t t0, uint64_t t1);
//void sim_region_timings(t_simulation *sim);
void sim_report_grid_zdf(t_simulation *sim, enum report_grid_type type, const int coord);
void sim_report_spec_zdf(t_simulation *sim, const int species, const int rep_type, const int pha_nx[],
		const float pha_range[][2]);
void sim_report_csv(t_simulation *sim);


#endif
