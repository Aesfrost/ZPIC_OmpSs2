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
#include <GASPI.h>
#include <mpi.h>

#include "zpic.h"
#include "utilities.h"
#include "simulation.h"
#include "emf.h"
#include "current.h"
#include "particles.h"
#include "timer.h"

// Simulation parameters (naming scheme : <type>-<number of particles>-<grid size x>-<grid size y>.c)
#include "input/lwfa-4000-16M-2000-512.c"
//#include "input/lwfa-8000-32M-4000-2048.c"
// #include "input/weibel-1000-604M-4096-4096.c"
//#include "input/weibel-500-151M-1024-1024.c"
// #include "input/weibel-500-67M-512-512.c"

#pragma oss assert("version.dependencies==regions")

int main(int argc, const char *argv[])
{
	if(argc != 2)
	{
		fprintf(stderr, "Please specify the number of regions");
		exit(1);
	}

	MPI_Init(&argc, &argv);
	CHECK_GASPI_ERROR(gaspi_proc_init(GASPI_BLOCK));

	// Initialize simulation
	t_simulation sim;
	sim_init(&sim, atoi(argv[1]));

	CHECK_GASPI_ERROR(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

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
		if(sim.proc_rank == ROOT)
			fprintf(stderr, "n = %i, t = %f\n", n, t);

		if (report(n, sim.ndump))
		{
#ifdef ENABLE_TASKING
			#pragma oss taskwait
			gaspi_flush_all_queues();
			#pragma oss taskwait
#endif
// 			sim_report(&sim);
		}

		sim_iter(&sim);
	}

#ifdef ENABLE_TASKING
	#pragma oss taskwait
	gaspi_flush_all_queues();
	#pragma oss taskwait
#endif

	CHECK_GASPI_ERROR(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

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

	CHECK_GASPI_ERROR(gaspi_proc_term(GASPI_BLOCK));
	MPI_Finalize();

	return 0;
}
