# ZPIC

[ZPIC](https://github.com/ricardo-fonseca/zpic) is a sequential 2D EM-PIC kinetic plasma simulator based on OSIRIS [1], implementing the same core algorithm and features. From ZPIC code, we developed several parallel versions to explore task-based programming models ([OmpSs-2](https://pm.bsc.es/ompss-2)) and emerging platforms (GPUs with [OpenACC](https://www.openacc.org/)). 

## Parallel versions

### OmpSs:
- Spatial row-wise decomposition (i.e., simulation split into regions alongside the y axis)
- All simulation steps defined as tasks
- Tasks are synchronized exclusively by data dependencies
- Fully asynchronous execution
- Local buffers (one per region) + parallel reduction for solving data races in the current deposition

### OpenACC:
- Target architecture: NVIDIA GPUs
- Spatial row-wise decomposition (i.e., simulation split into regions alongside the y axis). Each region is further divided into tiles (16x16 cells). 
- Particles: Structure of Arrays (SoA) for coalesced memory accesses
- Highly optimized particle advance (shared memory usage, atomic operations with infrequent memory conflicts, etc.)
- Highly optimized bucket sort based on [2, 3]
- NVIDIA Unified Memory + Explicit memory management for critical sections
- Support for multi-GPUs systems (OpenMP as management layer: launching kernels, synchronizing devices, etc.)

### OmpSs + OpenACC:
- Target architecture: NVIDIA GPUs
- Spatial row-wise decomposition (i.e., simulation split into regions alongside the y axis). Each region is further divided into tiles (16x16 cells). 
- Particles: Structure of Arrays (SoA) for coalesced memory accesses
- Highly optimized particle advance (shared memory usage, atomic operations with infrequent memory conflicts, etc.)
- Highly optimized bucket sort based on [2, 3]
- NVIDIA Unified Memory + Explicit memory management for critical sections
- Support for multi-GPUs systems
- OpenACC kernels incorporated as OmpSs tasks
- Asynchronous queues/streams for kernel overlapping
- Fully asynchronous execution
- Variant: Manual - Manual management of asynchronous queues and tasks (instead of entrusting this function to the NANOS6 runtime)

## Plasma Experiments / Input
Please check for the [ZPIC documentation](https://github.com/ricardo-fonseca/zpic/blob/master/doc/Documentation.md) for more information for setting up the simulation parameters. Included experiments: LWFA and Weibel Instability. For organization purpose, the simulation parameters are included on the file name with the following naming scheme: 
```
experiment type - number of time steps - number of particles per species - grid size x - grid size y
```

## Output

Like the original ZPIC, all versions report the simulation parameters in the ZDF format. For more information, please visit the [ZDF repository](https://github.com/ricardo-fonseca/zpic/tree/master/zdf).

## Compilation and Execution

### Compilation requirements:

OmpSs-based versions:
- [Nanos6 Runtime](https://github.com/bsc-pm/nanos6)
- [Mercurium Compiler](https://github.com/bsc-pm/mcxx)

OpenACC:
- PGI Compiler 19.10 or greater (later renamed as NVIDIA HPC SDK)

### OmpSs/OpenACC:
```
make
./zpic <Number of Regions>
```

### OmpSs@OpenAcc:
```
make affinity
./zpic <Number of Regions> <Percentage of regions dedicated to GPU> <Number of GPU regions>
```


## References

[1] R. A. Fonseca et al., ‘OSIRIS: A Three-Dimensional, Fully Relativistic Particle in Cell Code for Modeling Plasma Based Accelerators’, in Computational Science — ICCS 2002, Berlin, Heidelberg, 2002, vol. 2331, pp. 342–351. doi: 10.1007/3-540-47789-6_36.
[2]A. Jocksch, F. Hariri, T. M. Tran, S. Brunner, C. Gheller, and L. Villard, ‘A bucket sort algorithm for the particle-in-cell method on manycore architectures’, in Parallel Processing and Applied Mathematics, 2016, pp. 43–52. doi: 10.1007/978-3-319-32149-3_5.
[3]F. Hariri et al., ‘A portable platform for accelerated PIC codes and its application to GPUs using OpenACC’, Computer Physics Communications, vol. 207, pp. 69–82, Oct. 2016, doi: 10.1016/j.cpc.2016.05.008.

