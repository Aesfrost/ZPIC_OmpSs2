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
#include <omp.h>

#include "zpic.h"
#include "simulation.h"
#include "emf.h"
#include "current.h"
#include "particles.h"
#include "timer.h"
#include "utilities.h"

// Simulation parameters (naming scheme : <type>-<number of particles>-<grid size x>-<grid size y>.c)

/* Strong scaling */
#include "input/weibel-2000-151M-2048-2048.c"
//#include "input/lwfa-8000-74M-4000-2048.c"

/* Weak scaling */
//#include "input/weak/weibel-2000-151M-2048-2048.c" /* 1 GPU  */
//#include "input/weak/weibel-2000-303M-2900-2900.c" /* 2 GPUs */
//#include "input/weak/weibel-2000-467M-3600-3600.c" /* 3 GPUs */
//#include "input/weak/weibel-2000-604M-4096-4096.c" /* 4 GPUs */

int main(int argc, const char *argv[])
{
	if(argc != 2)
	{
		fprintf(stderr, "Wrong arguments. Expected: <number of regions>");
		exit(1);
	}

	// Initialize simulation
	t_simulation sim;
	sim_init(&sim, atoi(argv[1]));

	// Run simulation
	int n;
	float t;
	const int num_threads = MIN_VALUE(atoi(argv[1]), omp_get_max_threads());

#ifndef TEST
	fprintf(stderr, "Starting simulation ...\n\n");
#endif

	uint64_t t0, t1;
	t0 = timer_ticks();

	#pragma omp parallel num_threads(num_threads) private(n, t)
	{
		for (n = 0, t = 0.0; t <= sim.tmax; n++, t = n * sim.dt)
		{
#ifndef TEST
			#pragma omp master
			{
				fprintf(stderr, "n = %i, t = %f\n", n, t);
 				if (report(n, sim.ndump)) sim_report(&sim);
				sim.iter++;
			}

			#pragma omp barrier
#endif
			sim_iter(&sim);
		}
	}
	
	t1 = timer_ticks();

#ifndef TEST
	fprintf(stderr, "\nSimulation ended.\n\n");
#endif

	// Simulation times
	sim_timings(&sim, t0, t1, sim.tmax / sim.dt);

	// Cleanup data
	sim_delete(&sim);

	return 0;
}
