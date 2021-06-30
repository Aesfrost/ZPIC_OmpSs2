# Parallel ZPIC

[ZPIC](https://github.com/ricardo-fonseca/zpic) is a sequential 2D EM-PIC kinetic plasma simulator based on OSIRIS [1], implementing the same core algorithm and features. From the ZPIC code (em2d variant), we developed several parallel versions to explore tasking ([OmpSs-2](https://pm.bsc.es/ompss-2)) and emerging platforms (GPUs with [OpenACC](https://www.openacc.org/)). 

## Parallelization Strategy and Features

### General Strategy

In all parallel versions, the simulation space is split into multiple regions alongside the y axis (i.e., a row-wise decomposition). Each region stores both the particles inside it and the fraction of the grid they interact with, allowing both the particle advance and field integration to be performed locally. However, particles can exit their associated regions and must be transferred to their new location. Each region must also be padded with ghost cells, so that the thread processing the region can access grid quantities (current and EM fields) outside its boundaries. With this decomposition, the ZPIC code becomes:

```
function particle_advance(region)
	for each particle in region do
		Ep, Bp = interpolate_EMF(region.E, region.B, particle.pos)
		update_particle_momentum(Ep, Bp, particle.u) 
		old_pos = particle.pos
		particle_push(particle)
		deposit_current(region.J, old_pos, particle.pos)
		check_exiting_part(particle, region.outgoing_part)
	endfor 
	
	if moving_window is enable then
		shift_part_left(region.particles)
		inject_new_particles(region)
	endif
endfunction

for each time_step do
	for each region in simulation parallel do 	
		current_zero(region.J)
		particle_advance(region)
		add_incoming_part(region.particles, region.incoming_part)
		update_gc_add(region.J, region.neighbours.J)

		if region.J_filter is enable then 
			apply_filter(region.J, region.J_filter)
			update_gc_copy(region.J, region.neighbours.J) 
		endif 
		
		yee_b(region.E, region.B, dt / 2, dx, dy)
		yee_e(region.E, region.B, region.J, dt, dx, dy)
		yee_b(region.E, region.B, dt / 2, dx, dy)
					
		if moving_window is enable then
			shift_EMF_left(region.E, region.B)
		endif
		
		update_gc_copy(region.E, region.neighbours.E)
		update_gc_copy(region.B, region.neighbours.B)
	endfor
endfor

```
For a more detailed explanation, please check our upcoming paper in EuroPar2021. The pre-print version is available in [ArXiv](https://arxiv.org/abs/2106.12485). The `ompss2` version in this repository corresponds to the `zpic-reduction-async` variant in the paper.

### NVIDIA GPUs (OpenACC)
In addition to the spatial decomposition (see General Strategy), the particles within each region are sorted by tiles (16x16 cells) in order to use the Shared Memory as an explicit managed cache. During the particle advance, each tile is mapped to one SM. The SM then loads the local EM fields into the local memory, advances all the particles within, deposits atomically the current generated in a local buffer, and finally updates atomically electric current in region with the local values. This process is repeated for all tiles in a given region [2, 3]. Every time step, the program executes a highly optimized Bucket Sort (adapted from [3, 4]) to rearrange the particles, preserving data locality (which associates the particles with the tile their located in). The particles are stored as a Structure of Arrays (SoA) for accessing the global memory in coalesced fashion.

All kernels are developed in [OpenACC](https://www.openacc.org/) and the program relies on the NVIDIA Unified Memory for transferring data between the host and device. Some critical routines (e.g., particle sorting) uses explicit memory management. All programs supports multi-GPU systems.

### Features:
#### OmpSs-2:
- Each simulation step is defined as an OmpSs-2 tasks
- All tasks are synchronized through data dependencies
- Fully asynchronous execution
- Dynamic load balancing (overdecomposition + dynamic task scheduling)

#### OpenACC:
- Uses OpenMP for launching kernels in multiple devices, synchronizing their execution, etc.
- Prefetch routines to move data between devices before kernel execution.

#### OmpSs-2 + OpenACC:
- Uses OmpSs-2 for launching kernels in multiple devices, synchronizing their execution, etc.
- OpenACC kernels incorporated in OmpSs tasks
- Asynchronous queues/streams for kernel overlapping
- Fully asynchronous execution
- (Deprecated) Hybrid execution (CPU + GPU)

## Plasma Experiments / Input
Please check for the [ZPIC documentation](https://github.com/ricardo-fonseca/zpic/blob/master/doc/Documentation.md) for more information for setting up the simulation parameters. The LWFA (Laser Wakefield Acceleration) and Weibel (Instability) simulations are already included in all versions.

For organization purpose, each file is named after the simulation parameters according to the following scheme:
```
<experiment type> - <number of time steps> - <number of particles per species> - <grid size x> - <grid size y>
```

## Output

Like the original ZPIC, all versions report the simulation parameters in the ZDF format. For more information, please visit the [ZDF repository](https://github.com/ricardo-fonseca/zpic/tree/master/zdf).

The simulation timing and relevant information are displayed in the terminal after the simulation is completed.

## Compilation and Execution

### Requirements:

OmpSs-based versions:
- [Nanos6 Runtime](https://github.com/bsc-pm/nanos6)
- [Mercurium Compiler](https://github.com/bsc-pm/mcxx)

OpenACC:
- PGI Compiler 19.10 or newer (later renamed as NVIDIA HPC SDK)
- CUDA v9.0 or newer
- Pascal or newer GPUs
- With OmpSs-2, use this experimental version of the [Nanos6 Runtime](https://github.com/epeec/nanos6-openacc) (get-queue-affinity branch)


### Compilation Flags

`-DTEST`: Print the simulation timing and other information in a CSV friendly format. Disable all reporting and other terminal outputs

`-DENABLE_ADVISE` (`ON` by default): Enable CUDA MemAdvise routines to guide the Unified Memory System. All OpenACC versions

`-DENABLE_PREFETCH` (or `make prefetch`): Enable CUDA MemPrefetch routines (experimental). Pure OpenACC only.

`-DENABLE_AFFINITY` (or `make affinity`): Enable the use of device affinity (the runtime schedule openacc tasks based on the data location). Otherwise, Nanos6 runtime only uses 1 GPU. Only supported by OmpSs@OpenACC


### Commands

```
make <option> -j8
./zpic <number of regions>
```

## References

[1] R. A. Fonseca et al., ‘OSIRIS: A Three-Dimensional, Fully Relativistic Particle in Cell Code for Modeling Plasma Based Accelerators’, in Computational Science — ICCS 2002, Berlin, Heidelberg, 2002, vol. 2331, pp. 342–351. doi: 10.1007/3-540-47789-6_36.

[2] K. Germaschewski et al., ‘The Plasma Simulation Code: A modern particle-in-cell code with load-balancing and GPU support’, arXiv:1310.7866 [physics], Nov. 2015, Accessed: Nov. 25, 2019. [Online]. Available: [http://arxiv.org/abs/1310.7866](http://arxiv.org/abs/1310.7866)

[3] A. Jocksch, F. Hariri, T. M. Tran, S. Brunner, C. Gheller, and L. Villard, ‘A bucket sort algorithm for the particle-in-cell method on manycore architectures’, in Parallel Processing and Applied Mathematics, 2016, pp. 43–52. doi: 10.1007/978-3-319-32149-3_5.

[4] F. Hariri et al., ‘A portable platform for accelerated PIC codes and its application to GPUs using OpenACC’, Computer Physics Communications, vol. 207, pp. 69–82, Oct. 2016, doi: 10.1016/j.cpc.2016.05.008.


