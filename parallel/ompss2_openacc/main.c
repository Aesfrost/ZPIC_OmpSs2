/*
 Copyright (C) 2017 Instituto Superior Tecnico

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
 */

#include <stdio.h>
#include <stdlib.h>

#include "zpic.h"
#include "simulation.h"
#include "emf.h"
#include "current.h"
#include "particles.h"
#include "timer.h"

// Simulation parameters (naming scheme : <type>-<number of particles>-<grid size x>-<grid size y>.c)
#include "input/weibel-500-67M-512-512.c"

int main(int argc, const char *argv[])
{
	if(argc != 4)
	{
		fprintf(stderr, "Wrong arguments. Expected: <number of regions> <percentage of regions using GPU> "
				"<number of GPU regions>");
		exit(1);
	}

	// Initialize simulation
	t_simulation sim;
	sim_init(&sim, atoi(argv[1]), atof(argv[2]), atoi(argv[3]));

	// Run simulation
	int n;
	float t;

#ifdef MANUAL_GPU_SETUP
	printf("Manual Setup...\n");
#endif

	fprintf(stderr, "Starting simulation ...\n\n");

	uint64_t t0, t1;
	t0 = timer_ticks();

	for (n = 0, t = 0.0; t <= sim.tmax; n++, t = n * sim.dt)
	{
//		if(n == 50) break;

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

	//sim_report(&sim);

	t1 = timer_ticks();
	fprintf(stderr, "\nSimulation ended.\n\n");

	// Simulation times
	sim_timings(&sim, t0, t1, n);

	// Cleanup data
	sim_delete(&sim);

	return 0;
}
