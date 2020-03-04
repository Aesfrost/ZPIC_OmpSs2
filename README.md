# ZPIC - OmpSs-2 

ZPIC is a 2D plasma simulator using the widely used PIC (particle-in-cell) algorithm. The program uses a finite difference model to simulate eletromagnetic plasma events. These version is an adaptation of ZPIC to be executed in parallel using the OmpSs-2 programming model. The original serial code belongs to the [ZPIC suite](https://github.com/ricardo-fonseca/zpic).  

## Features (Parallel Version)
- Rows decomposition
- Parallelism based on tasks and data dependencies (OmpSs-2)
- No taskwait (or synchronism) between iterations of the simulation (OmpSs-2)

## Unsupported Features
In the current version, the parallel ZPIC doesn't support:
- Current smoothing in the y direction
- ZDF file format 

## Plasma Experiments / Input
In the same way of the original code, the simulation paramenters are set in a .c file in input folder that are later included in the main.c
```
#include "input/<filename>.c"
```
Two widely known plasma experiments - LWFA and Weibel Instability - are already included. Each experiment have a smaller and a larger variant.

## Output
The same file used to set the parameters of the simulation, defines the ouput files and the frequency of the output (in terms of simulation iterations).
By default, the ZPIC produce the following maps (every 50 iterations):
- Charge (for each particle type)
- Eletric field magnitude
- Magnetic field magnitude

Also, ZPIC monitors the EM fields' energy and particles' energy through all the simulation.

All output files are in the .csv format with ";" as delimiter. The last column of each map **MUST** be removed before being processed!

## Compilation and Execution
```
make
./zpic <Number of Regions>
```

Compilation requirements:
- [Nanos6 Runtime](https://github.com/bsc-pm/nanos6)
- [Mercurium Compiler](https://github.com/bsc-pm/mcxx)
