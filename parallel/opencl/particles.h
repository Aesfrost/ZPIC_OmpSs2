/*********************************************************************************************
 ZPIC
 particles.h

 Created by Ricardo Fonseca on 11/8/10.
 Modified by Nicolas Guidotti on 11/06/20

 Copyright 2010 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#ifndef __PARTICLES__
#define __PARTICLES__

#include "zpic.h"
#include "emf.h"
#include "current.h"

#define TILE_SIZE 16
#define MAX_SPNAME_LEN 32
#define MAX_LEAVING_PART 0.25 // Percentage of the total number of particles

//typedef struct {
//	int ix, iy;
//	t_part_data x, y;
//	t_part_data ux, uy, uz;
//} t_part;

enum density_type {
	UNIFORM, STEP, SLAB
};

typedef struct {
	float n;				// reference density (defaults to 1.0, multiplies density profile)
	enum density_type type;		// Density profile type
	float start, end;		// Position of the plasma start/end, in simulation units

} t_density;

typedef struct {
	cl_int2 *cell_idx;
	cl_float2 *position;
	cl_float3 *velocity;

	int np;
	int np_max;
} t_part_vector;

typedef struct {
	char name[MAX_SPNAME_LEN];

	// Particle data buffer
//	t_part *part;
//	int np;
//	int np_max;

	t_part_vector part_vector;

	t_part_vector incoming_part;

	// mass over charge ratio
	t_part_data m_q;

	// total kinetic energy
	double energy;

	// charge of individual particle
	t_part_data q;

	// Number of particles per cell
	int ppc[2];

	// Density profile to inject
	t_density density;

	// Initial momentum of particles
	t_part_data ufl[3];
	t_part_data uth[3];

	// Simulation box info
	int nx[2];
	t_part_data dx[2];
	t_part_data box[2];

	// Time step
	float dt;

	// Iteration number
	int iter;

	// Moving window
	int moving_window;
	int n_move;

	cl_int2 n_tiles;
	int *tile_offset;
	int *np_per_tile;

	t_part_vector temp_part;
	int *sort_counter;
	int *target_idx;

} t_species;

void spec_new(t_species *spec, char name[], const t_part_data m_q, const int ppc[],
        const t_part_data ufl[], const t_part_data uth[], const int nx[], t_part_data box[],
        const float dt, t_density *density);
void spec_init_tiles(t_species *spec, const int nx[2]);
void spec_delete(t_species *spec);
void spec_advance(t_species *spec, t_emf *emf, t_current *current);

void spec_set_moving_window(t_species *spec);

double spec_time(void);
double spec_perf(void);

void spec_sort(t_part_vector *part_vector, t_part_vector *temp_part, t_part_vector *new_part,
        int *restrict tile_offset, int *restrict np_per_tile, int *restrict sort_counter,
        int *target_idx, const cl_int2 n_tiles, const int nx[2], const int moving_window,
        const int shift, const int ppc[2]);

// OpenCL Kernels
#pragma omp target device(opencl) copy_deps ndrange(2, n_tiles.x * 512, n_tiles.y, 512, 1) file(kernel_gpu.cl)
#pragma omp task in(E_buf[0; field_size]) in(B_buf[0; field_size]) inout(part_cell_idx[0; np_max]) \
	inout(part_positions[0; np_max]) inout(part_velocities[0; np_max]) inout(J_buf[0; field_size]) \
	in(tile_offset[0: n_tiles.x * n_tiles.y]) inout(np_per_tile[0: n_tiles.x * n_tiles.y])
void spec_advance_opencl(cl_int2 *restrict part_cell_idx, cl_float2 *restrict part_positions,
        cl_float3 *restrict part_velocities, const int *restrict tile_offset,
        int *restrict np_per_tile, const int np_max, const cl_float3 *restrict E_buf,
        const cl_float3 *restrict B_buf, cl_float3 *restrict J_buf, const int nrow,
        const int field_size, const t_part_data tem, const t_part_data dt_dx,
        const t_part_data dt_dy, const t_part_data qnx, const t_part_data qny, const t_part_data q,
        const int nx0, const int nx1, const cl_int2 n_tiles, const int moving_window,
        const int shift);

#pragma omp target device(opencl) copy_deps ndrange(2, n_tiles.x * 128, n_tiles.y, 128, 1) file(kernel_gpu.cl)
#pragma omp task in(part_cell_idx[0; np_max]) in(part_positions[0; np_max]) in(part_velocities[0; np_max]) \
		out(temp_cell_idx[0; sort_size]) out(temp_positions[0; sort_size]) \
		out(temp_velocities[0; sort_size]) out(target_idx[0; sort_size]) \
		inout(counter[0; n_tiles.x * n_tiles.y]) in(tile_offset[0: n_tiles.x * n_tiles.y])
void spec_sort_1(const cl_int2 *restrict part_cell_idx, const cl_float2 *restrict part_positions,
        const cl_float3 *restrict part_velocities, cl_int2 *restrict temp_cell_idx,
        cl_float2 *restrict temp_positions, cl_float3 *restrict temp_velocities,
        int *restrict target_idx, int *restrict counter, const int *restrict tile_offset,
        const cl_int2 n_tiles, const int max_holes_per_tile, const int np, const int np_max,
        const int sort_size, const int nx0);

#pragma omp target device(opencl) copy_deps ndrange(1, np_inj, 128) file(kernel_gpu.cl)
#pragma omp task in(new_cell_idx[0; np_inj]) in(new_positions[0; np_inj]) in(new_velocities[0; np_inj]) \
		inout(temp_cell_idx[0; np]) inout(temp_positions[0; np]) inout(temp_velocities[0; np]) \
		inout(counter[0; n_tiles.x * n_tiles.y])
void spec_sort_1_mw(cl_int2 *restrict temp_cell_idx, cl_float2 *restrict temp_positions,
        cl_float3 *restrict temp_velocities, const cl_int2 *restrict new_cell_idx,
        const cl_float2 *restrict new_positions, const cl_float3 *restrict new_velocities,
        int *restrict counter, const int np, const int np_inj, const cl_int2 n_tiles);

#pragma omp target device(opencl) copy_deps ndrange(1, 128 * n_tiles.x * n_tiles.y, 128) file(kernel_gpu.cl)
#pragma omp task inout(part_cell_idx[0; np_max]) inout(part_positions[0; np_max]) inout(part_velocities[0; np_max]) \
		in(temp_cell_idx[0; sort_size]) in(temp_positions[0; sort_size]) in(temp_velocities[0; sort_size]) \
		in(target_idx[0; sort_size]) in(counter[0; n_tiles.x * n_tiles.y])
void spec_sort_2(cl_int2 *restrict part_cell_idx, cl_float2 *restrict part_positions,
        cl_float3 *restrict part_velocities, const cl_int2 *restrict temp_cell_idx,
        const cl_float2 *restrict temp_positions, const cl_float3 *restrict temp_velocities,
        const int *restrict target_idx, const int *counter, const cl_int2 n_tiles,
        const int max_holes_per_tile, const int sort_size, const int np_max);

/*********************************************************************************************
 Diagnostics
 *********************************************************************************************/

#define CHARGE 		0x1000
#define PHA    		0x2000
#define PARTICLES   0x3000
#define X1     		0x0001
#define X2     		0x0002
#define U1     		0x0004
#define U2     		0x0005
#define U3     		0x0006

#define PHASESPACE(a,b) ((a) + (b)*16 + PHA)

void spec_deposit_pha(const t_species *spec, const int rep_type, const int pha_nx[],
        const float pha_range[][2], float *buf);

void spec_report(const t_species *spec, const int rep_type, const int pha_nx[],
        const float pha_range[][2], const char path[128]);

void spec_deposit_charge(const t_species *spec, float *charge);
void spec_report_csv(const t_species *spec, const char sim_name[64]);
void spec_calculate_energy(t_species *restrict spec);

#endif
