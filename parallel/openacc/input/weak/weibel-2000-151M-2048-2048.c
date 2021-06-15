/**
 * ZPIC - em2d
 *
 * Weibel instability
 */

#include <stdlib.h>
#include "../../simulation.h"

void sim_init(t_simulation *sim, int n_regions)
{
	// Time step
	float dt = 0.035;
	float tmax = 70.0;

	// Simulation box
	int nx[2] = {2048, 2048};
	float box[2] = {102.4, 102.4};

	// Diagnostic frequency
	int ndump = 500;

	// Initialize particles
	const int n_species = 2;
	t_species *species = (t_species*) malloc(n_species * sizeof(t_species));

	// Use 8x8 particles per cell
	int ppc[] = {6, 6};

	// Initial fluid and thermal velocities
	t_part_data ufl[] = {0.0, 0.0, 0.6};
	t_part_data uth[] = {0.1, 0.1, 0.1};

	spec_new(&species[0], "electrons", -1.0, ppc, ufl, uth, nx, box, dt, NULL, nx[1]);

	ufl[2] = -ufl[2];
	spec_new(&species[1], "positrons", +1.0, ppc, ufl, uth, nx, box, dt, NULL, nx[1]);

	// Initialize Simulation data
	sim_new(sim, nx, box, dt, tmax, ndump, species, n_species, "weibel-2000-151M-2048-2048", n_regions);

	free(species);
}

void sim_report(t_simulation *sim)
{
	//sim_report_csv(sim);
	sim_report_energy(sim);

	// Bx, By, Bz
	sim_report_grid_zdf(sim, REPORT_BFLD, 0);
	sim_report_grid_zdf(sim, REPORT_BFLD, 1);
	sim_report_grid_zdf(sim, REPORT_BFLD, 2);

	// Jz
	sim_report_grid_zdf(sim, REPORT_CURRENT, 2);

	// electron and positron density
	sim_report_spec_zdf(sim, 0, CHARGE, NULL, NULL);
	sim_report_spec_zdf(sim, 1, CHARGE, NULL, NULL);
}
