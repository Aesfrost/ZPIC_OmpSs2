#ifndef __SIMULATION__
#define __SIMULATION__

#include <stdint.h>

#include "region.h"
#include "particles.h"
#include "emf.h"
#include "current.h"

typedef struct {
	char name[64];

	// Time step
	float dt;
	float tmax;

	// Diagnostic frequency
	int ndump;

	// Simulation data
	int nx[2];
	bool moving_window;
	t_region *first_region; // Pointer to the first region (id = 0)

	int iter;

} t_simulation;

void sim_new(t_simulation *sim, int nx[], float box[], float dt, float tmax, int ndump, t_species *species,
		int n_species, char name[64], int n_regions);
void sim_init(t_simulation *sim, int n_regions);
void sim_set_moving_window(t_simulation *sim);
void sim_set_smooth(t_simulation *sim, t_smooth *smooth);
void sim_add_laser(t_simulation *sim, t_emf_laser *laser);
void sim_delete(t_simulation *sim);

void sim_iter(t_simulation *sim);

int report(int n, int ndump);
void sim_report(t_simulation *sim);
void sim_report_energy(t_simulation *sim);
void sim_report_charge(t_simulation *sim);
void sim_report_emf(t_simulation *sim);
void sim_timings(t_simulation *sim, uint64_t t0, uint64_t t1);

#endif
