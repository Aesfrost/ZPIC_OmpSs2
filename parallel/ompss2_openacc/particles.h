/*********************************************************************************************
 ZPIC
 particles.h

 Created by Ricardo Fonseca on 11/8/10.
 Modified by Nicolas Guidotti on 14/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/


#ifndef __PARTICLES__
#define __PARTICLES__

#include <stdbool.h>

#include "zpic.h"
#include "emf.h"
#include "current.h"

#define BIN_SIZE 16
#define MAX_SPNAME_LEN 32

typedef struct {
	int ix, iy;
	t_part_data x, y;
	t_part_data ux, uy, uz;

	bool safe_to_delete; // Mark the particle as invalid (the particle exited the region)

} t_part;

enum density_type {
	UNIFORM, STEP, SLAB
};

// SoA = Structure of Arrays / AoS = Array of Structures
enum vector_type {
	SoA, AoS
};

typedef struct {
	float n;				// reference density (defaults to 1.0, multiplies density profile)
	enum density_type type;		// Density profile type
	float start, end;		// Position of the plasma start/end, in simulation units

} t_density;

// Particle data buffer
typedef struct {

	// AoS
	t_part *part;

	// SoA
	int *ix, *iy;
	t_part_data *x, *y;
	t_part_data *ux, *uy, *uz;
	bool *safe_to_delete;

	enum vector_type type;

	int size;
	int size_max;

} t_particle_vector;

typedef struct {
	char name[MAX_SPNAME_LEN];

	// Particle data buffer (CPU)
	t_particle_vector main_vector;
	t_particle_vector temp_buffer[2];    // Temporary buffer for incoming particles

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
	bool moving_window;
	int n_move;

	// Sort
	int n_bins_x;
	int n_bins_y;
	int *bin_offset;

} t_species;

// Setup
void spec_new(t_species *spec, char name[], const t_part_data m_q, const int ppc[],
		const t_part_data ufl[], const t_part_data uth[], const int nx[], t_part_data box[],
		const float dt, t_density *density, const int region_size);
void spec_inject_particles(t_species *spec, const int range[][2]);
void spec_delete(t_species *spec);

// Utilities
void convert_vector(t_particle_vector *restrict vector, enum vector_type final_type);

// Report - General
double spec_time(void);
double spec_perf(void);

// CPU Tasks
#pragma oss task label(Spec Advance) \
	in(emf->E_buf[0; emf->total_size]) in(emf->B_buf[0; emf->total_size]) \
	inout(spec->main_vector) inout(current->J_buf[0; current->total_size]) priority(5)
void spec_advance(t_species *spec, t_emf *emf, t_current *current, int limits_y[2]);

#pragma oss task inout(spec->main_vector) out(lower_spec->temp_buffer[1]) \
	out(upper_spec->temp_buffer[0]) label(Spec PP)
void spec_post_processing(t_species *spec, t_species *upper_spec, t_species *lower_spec,
		int limits_y[2]);

#pragma oss task in(spec->temp_buffer[0:1]) inout(spec->main_vector) label(Spec Update)
void spec_update_main_vector(t_species *spec);

// OpenAcc Tasks
#pragma oss task label(Spec Kernel (GPU)) device(openacc) \
	in(emf->E_buf[0; emf->total_size]) in(emf->B_buf[0; emf->total_size]) \
	inout(current->J_buf[0; current->total_size]) inout(spec->main_vector) priority(5)
void spec_advance_openacc(t_species *restrict const spec, const t_emf *restrict const emf,
		t_current *restrict const current, const int limits_y[2]); // Advance a single species (openacc)

#pragma oss task label(Spec Post Processing (GPU)) device(openacc) \
	inout(spec->main_vector) out(lower_spec->temp_buffer[1]) out(upper_spec->temp_buffer[0])
void spec_post_processing_1_openacc(t_species *restrict spec, t_species *restrict const upper_spec,
		t_species *restrict const lower_spec, const int limits_y[2]); // Post processing for a single species (openacc)

#pragma oss task label(Spec Update (GPU)) in(spec->temp_buffer[0:1]) inout(spec->main_vector) //device(openacc)
void spec_post_processing_2_openacc(t_species *restrict spec, const int limits_y[2]);

#pragma oss task label(Spec Sort (GPU)) inout(spec->main_vector)
void spec_sort_openacc(t_species *spec, const int limits_y[2]);

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

// Phase space
void spec_deposit_pha(const t_species *spec, const int rep_type, const int pha_nx[],
		const float pha_range[][2], float *buf);
void spec_rep_pha(const t_part_data *buffer, const int rep_type, const int pha_nx[],
		const float pha_range[][2], const int iter_num, const float dt, const char path[128]);

// Charge map
void spec_deposit_charge(const t_species *spec, float *charge);
void spec_rep_charge(t_part_data *restrict charge, const int true_nx[2], const t_fld box[2],
		const int iter_num, const float dt, const bool moving_window, const char path[128]);

// Energy
void spec_calculate_energy(t_species *spec);

#endif
