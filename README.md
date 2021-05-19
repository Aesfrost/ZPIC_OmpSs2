# ZPIC - OmpSs-2 

ZPIC is a 2D PIC (particle-in-cell) plasma simulator that uses uses a finite difference model to simulate eletromagnetic events ([ZPIC suite](https://github.com/ricardo-fonseca/zpic)). This repository contains several parallel implementations of the ZPIC code following either OpenMP or [OmpSs-2](https://github.com/bsc-pm/ompss-2-releases) programming models.


## Parallel Versions

|         Version        |  Model | Decomposition |             Synchronization             | Data Race  Solution | Asynchronous? | Overdecomposition |
|:----------------------:|:------:|:-------------:|:---------------------------------------:|:-------------------:|:-------------:|:-----------------:|
|    zpic-parallel-for   | OpenMP |    Particle   |                 Barriers                |      Reduction      |       No      |         No        |
|      zpic-tasklike     | OpenMP | Spatial (Row) |                 Barriers                |      Reduction      |       No      |        Yes        |
|   zpic-reduction-sync  |  OmpSs | Spatial (Row) | Data Dependencies  (Barrier at the end) |      Reduction      |    Partial    |        Yes        |
|  zpic-commutative-sync |  OmpSs | Spatial (Row) | Data Dependencies  (Barrier at the end) |     Commutative     |    Partial    |        Yes        |
|  zpic-reduction-async  |  OmpSs | Spatial (Row) |            Data Dependencies            |      Reduction      |      Full     |        Yes        |
| zpic-commutative-async |  OmpSs | Spatial (Row) |            Data Dependencies            |     Commutative     |      Full     |        Yes        |

## Input / Output 

All parameters of the simulation are defined into a .c file within the input folder. Two widely known plasma experiments - LWFA and Weibel Instability - are already included as well as the simulation of two uniform plasmas (cold and warm). 

Like the original ZPIC, all versions generates reports into the ZDF file format. For more information, please visit the [ZDF repository](https://github.com/ricardo-fonseca/zpic/tree/master/zdf). Besides the ZDF files, the parallel ZPIC can also  produce .csv files (delimiter = ";") for some of the simulation parameters.

## Compilation and Execution

### OmpSs/OpenMP:
```
make
./zpic <Number of Regions>
```
### Serial:
(`zpic-parallel-for` follows the same guidelines as the serial version)
```
make
./zpic
```
### Compilation requirements:
- [Nanos6 Runtime](https://github.com/bsc-pm/nanos6)
- [Mercurium Compiler](https://github.com/bsc-pm/mcxx)
