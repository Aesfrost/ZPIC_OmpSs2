/*****************************************************************************************
 Copyright (C) 2020 Instituto Superior Tecnico

 This file is part of the ZPIC Educational code suite

 The ZPIC Educational code suite is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 The ZPIC Educational code suite is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with the ZPIC Educational code suite. If not, see <http://www.gnu.org/licenses/>.

 *****************************************************************************************
 The original ZPIC was modified to include the support for the OmpSs-2 programming model.
 *****************************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "zpic.h"
#include "simulation.h"
#include "emf.h"
#include "current.h"
#include "particles.h"
#include "timer.h"

// Simulation parameters (naming scheme : <type>-<number of particles>-<grid size x>-<grid size y>.c)
//#include "input/lwfa-4000-16M-2000-512.c"
#include "input/weibel-500-67M-512-512.c"

int main(int argc, const char *argv[])
{
	if(argc != 2)
	{
		fprintf(stderr, "Please specify the number of regions");
		exit(1);
	}

	// Initialize simulation
	t_simulation sim;
	sim_init(&sim, atoi(argv[1]));

	// Run simulation
	int n;
	float t;

	fprintf(stderr, "Starting simulation ...\n\n");

	uint64_t t0, t1;
	t0 = timer_ticks();

	for (n = 0, t = 0.0; t <= sim.tmax; n++, t = n * sim.dt)
	{
//		if(n == 2) break;

//		fprintf(stderr, "n = %i, t = %f\n", n, t);
//
//		if (report(n, sim.ndump))
//		{
//			#pragma oss taskwait
//			sim_report(&sim);
//		}

		sim_iter(&sim);
	}

	#pragma oss taskwait

	t1 = timer_ticks();
	fprintf(stderr, "\nSimulation ended.\n\n");

	// Simulation times
	sim_timings(&sim, t0, t1);

	// Cleanup data
	sim_delete(&sim);

	return 0;
}
