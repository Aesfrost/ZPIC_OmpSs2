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
#include "utilities.h"
#include "simulation.h"
#include "emf.h"
#include "current.h"
#include "particles.h"
#include "timer.h"

// Simulation parameters (naming scheme : <type>-<number of particles>-<grid size x>-<grid size y>.c)
//#include "input/lwfa-4000-16M-2000-512.c"
//#include "input/lwfa-8000-32M-4000-2048.c"
//#include "input/weibel-1000-604M-2048-2048.c"
// #include "input/weibel-500-151M-1024-1024.c"

//#include "input/weak/warm-1n.c"
//#include "input/weak/warm-4n.c"
//#include "input/weak/warm-16n.c"
//#include "input/weak/warm-64n.c"
//#include "input/weak/warm-256n.c"

// #include "input/weak/weibel-1n.c"
//#include "input/weak/weibel-4n.c"
//#include "input/weak/weibel-16n.c"
//#include "input/weak/weibel-64n.c"
//#include "input/weak/weibel-256n.c"

#include "input/weak/cold-1n.c"
//#include "input/weak/cold-4n.c"
//#include "input/weak/cold-16n.c"
//#include "input/weak/cold-64n.c"
//#include "input/weak/cold-256n.c"

#pragma oss assert("version.dependencies==regions")

int main(int argc, const char *argv[])
{
	if(argc != 2)
	{
		fprintf(stderr, "Please specify the number of regions");
		exit(1);
	}

#ifdef ENABLE_TASKING
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

	if(provided != MPI_THREAD_MULTIPLE)
	{
		fprintf(stderr, "Error: MPI installation doesn't support multithreading\n");
		exit(1);
	}
#else
	MPI_Init(&argc, &argv);
#endif

	// Initialize simulation
	t_simulation sim;
	sim_init(&sim, atoi(argv[1]));
	CHECK_MPI_ERROR(MPI_Barrier(MPI_COMM_WORLD));

	// Run simulation
	int n;
	float t;

#ifndef TEST
	if(sim.proc_rank == ROOT)
		fprintf(stderr, "Starting simulation ...\n\n");
#endif

	uint64_t t0 = timer_ticks();

	for (n = 0, t = 0.0; t <= sim.tmax; n++, t = n * sim.dt)
	{
//		if(sim.proc_rank == ROOT)
//			fprintf(stderr, "n = %i, t = %f\n", n, t);
//
//		if (report(n, sim.ndump))
//		{
//#ifdef ENABLE_TASKING
//			#pragma oss taskwait
//#endif
//			sim_report(&sim);
//		}

		sim_iter(&sim);
	}

#ifdef ENABLE_TASKING
	#pragma oss taskwait
#endif

	CHECK_MPI_ERROR(MPI_Barrier(MPI_COMM_WORLD));

	if(sim.proc_rank == ROOT)
	{
#ifndef TEST
		fprintf(stderr, "\nSimulation ended.\n\n");
#endif

		// Simulation times
		sim_timings(&sim, t0, timer_ticks());
	}

	// Cleanup data
	sim_delete(&sim);
	MPI_Finalize();

	return 0;
}
