/**
 * ZPIC - em2d
 *
 * Laser Wakefield Acceleration
 */

#include <stdlib.h>
#include <math.h>

#include "../simulation.h"

void sim_init(t_simulation *sim)
{

	// Time step
	float dt = 0.014;
	float tmax = 28;  //20.314;

	// Simulation box
	int nx[2] = {2000, 256};
	float box[2] = {40.0, 51.2};

	// Diagnostic frequency
	int ndump = 50;

	// Initialize particles
	const int n_species = 1;

	// Use 4x2 particles per cell
	int ppc[] = {4, 2};

	// Density profile
	t_density density = {.type = STEP, .start = 20.0};

	t_species *species = (t_species*) malloc(n_species * sizeof(t_species));
	spec_new(&species[0], "electrons", -1.0, ppc, NULL, NULL, nx, box, dt, &density);

	// Initialize Simulation data
	sim_new(sim, nx, box, dt, tmax, ndump, species, n_species, "lwfa-2000-4M-2000-256");

	// Add laser pulse (this must come after sim_new)
	t_emf_laser laser = {.type = GAUSSIAN, .start = 17.0, .fwhm = 2.0, .a0 = 2.0, .omega0 = 10.0, .W0 = 4.0,
			.focus = 20.0, .axis = 12.8, .polarization = M_PI_2};
	sim_add_laser(sim, &laser);

	// Set moving window (this must come after sim_new)
	sim_set_moving_window(sim);

	// Set current smoothing (this must come after sim_new)
	t_smooth smooth = {.xtype = COMPENSATED, .xlevel = 4};

	sim_set_smooth(sim, &smooth);
}

void sim_report(t_simulation *sim)
{
	//sim_report_csv(sim);
	sim_report_energy(sim);

	// Bx, By, Bz
	sim_report_grid_zdf(sim, REPORT_BFLD, 0);
	sim_report_grid_zdf(sim, REPORT_BFLD, 1);
	sim_report_grid_zdf(sim, REPORT_BFLD, 2);

	// All electric field components
	sim_report_grid_zdf(sim, REPORT_EFLD, 0);
	sim_report_grid_zdf(sim, REPORT_EFLD, 1);
	sim_report_grid_zdf(sim, REPORT_EFLD, 2);

	// Charge density
	sim_report_spec_zdf(sim, 0, CHARGE, NULL, NULL);

	// x1u1 phasespace
	const int pha_nx[] = {1024, 512};
	const float pha_range[][2] = { {0.0, 20.0}, {-2.0, +2.0}};
	sim_report_spec_zdf(sim, 0, PHASESPACE(X1, U1), pha_nx, pha_range);

}
