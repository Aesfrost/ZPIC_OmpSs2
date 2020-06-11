# ZPIC - OmpSs-2 

ZPIC is a 2D plasma simulator using the widely used PIC (particle-in-cell) algorithm. The program uses a finite difference model to simulate eletromagnetic plasma events. These version is an adaptation of ZPIC, so it can be executed in parallel using the OmpSs-2 programming model. The original serial code belongs to the [ZPIC suite](https://github.com/ricardo-fonseca/zpic).  

## Features 

### OmpSs:
- Rows decomposition
- Parallelism based on tasks and data dependencies (OmpSs-2)
- No taskwait (or synchronism) between iterations of the simulation (OmpSs-2)

### OmpSs@OpenAcc (under development):
- Rows decomposition
- Parallelism based on tasks and data dependencies (OmpSs-2)
- No taskwait (or synchronism) between iterations of the simulation (OmpSs-2)
- The OpenAcc kernels are integrated into OmpSs tasks
- Hybrid structure for the particles (Structure of Arrays (SoA) for GPU and Array of Structure (AoS) for CPU)
- Manual allocation of regions to be execute in the GPU (based on a percentage of the total number of regions)
- Bucket Sort every 15 iterations

### OpenAcc:
- Based on the serial version
- All computation are done in the GPU
- Particles use a Structure of Array (SoA) to improve GPU performance
- Bucket Sort every 15 iterations

## Plasma Experiments / Input
In the same way of the original code, the simulation paramenters are set in a .c file in input folder that are later included in the main.c

Two widely known plasma experiments - LWFA and Weibel Instability - are already included. Each experiment have a smaller and a larger variant.

## Output
The same file used to set the parameters of the simulation defines the ouput files and the frequency of the output (in terms of simulation iterations).

### ZDF Format
Like the original ZPIC, both serial and pure OmpSs-2 versions supports the ZDF file format. For more information, please visit the [ZDF repository](https://github.com/ricardo-fonseca/zpic/tree/master/zdf).

In the future, all the versions will have support for this file format.

### CSV Format
Besides the ZDF files, ZPIC can produce .csv files (delimiter = ";") for the following parameters:
- Charge map (for each particle type)
- Eletric field magnitude
- Magnetic field magnitude
- EM fields' energy
- Particles' energy

## Compilation and Execution
### OmpSs:
```
make
./zpic <Number of Regions>
```
### OpenAcc/Serial:
```
make
./zpic
```

### OmpSs@OpenAcc (under development):
```
make
./zpic <Number of Regions> <Percentage of regions dedicated to GPU> <Number of GPU regions>
```
### Compilation requirements:
- [Nanos6 Runtime](https://github.com/bsc-pm/nanos6)
- [Mercurium Compiler](https://github.com/bsc-pm/mcxx)
