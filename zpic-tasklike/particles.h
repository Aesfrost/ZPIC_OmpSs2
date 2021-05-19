/*********************************************************************************************
 ZPIC
 particles.h

 Created by Ricardo Fonseca on 11/8/10.
 Modified by Nicolas Guidotti on 11/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#ifndef __PARTICLES__
#define __PARTICLES__

#include <stdbool.h>
#include <stddef.h>

#include "zpic.h"
#include "emf.h"
#include "current.h"

#define MAX_SPNAME_LEN 32

typedef struct {
	int ix, iy;
	t_part_data x, y;
	t_part_data ux, uy, uz;

	// Mark the particle as invalid (the particle exited the region)
	bool safe_to_delete;

} t_part;

enum density_type {
	UNIFORM, STEP, SLAB
};

typedef struct {
	float n;				// reference density (defaults to 1.0, multiplies density profile)
	enum density_type type;		// Density profile type
	float start, end;		// Position of the plasma start/end, in simulation units

} t_density;

// Particle data buffer
typedef struct {
	t_part *data;
	int size;
	int size_max;

} t_particle_vector;

typedef struct {
	char name[MAX_SPNAME_LEN];

	// Particle data buffer
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
} t_species;

// Setup
void spec_new(t_species *spec, char name[], const t_part_data m_q, const int ppc[],
		const t_part_data ufl[], const t_part_data uth[], const int nx[], t_part_data box[],
		const float dt, t_density *density);
void spec_inject_particles(t_species *spec, const int range[][2]);
void spec_delete(t_species *spec);

// Report - General
double spec_time(void);
double spec_perf(void);

// Utilities
void realloc_vector(void **restrict ptr, const int old_size, const int new_size, const size_t type_size);

// CPU Tasks
void spec_advance(t_species *spec, t_emf *emf, t_current *current, int limits_y[2]);
void spec_post_processing(t_species *spec, t_species *upper_spec, t_species *lower_spec,
		int limits_y[2]);
void spec_update_main_vector(t_species *spec);

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
