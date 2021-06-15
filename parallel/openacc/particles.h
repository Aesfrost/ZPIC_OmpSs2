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

#define THREAD_BLOCK 320
#define MAX_SPNAME_LEN 32
#define EXTRA_NP 0.05 // Overallocation (fraction of the total)
#define MAX_LEAVING_PART 0.3 // Maximum percentage of particles that can exchanged between tiles

enum density_type {
	UNIFORM, STEP, SLAB
};

typedef struct {
	float n;				// reference density (defaults to 1.0, multiplies density profile)
	enum density_type type;		// Density profile type
	float start, end;		// Position of the plasma start/end, in simulation units

} t_density;

// Particle data buffer (SoA)
typedef struct {
	int *ix, *iy;
	t_part_data *x, *y;
	t_part_data *ux, *uy, *uz;
	bool *invalid;

	int size;
	int size_max;
	bool enable_vector;

} t_particle_vector;

typedef struct {
	char name[MAX_SPNAME_LEN];

	// Particle data buffer
	t_particle_vector main_vector;

	// Temporary buffer for incoming particles
	// 1 - From a lower region / 0 - From an upper region
	t_particle_vector incoming_part[3];

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

	// Outgoing particles
	// 0 - Going down / 1 - Going up
	t_particle_vector *outgoing_part[2];

	// Sort
	int n_tiles_x;
	int n_tiles_y;
	int *tile_offset;

} t_species;

// Setup
void spec_new(t_species *spec, char name[], const t_part_data m_q, const int ppc[],
		const t_part_data ufl[], const t_part_data uth[], const int nx[], t_part_data box[],
		const float dt, t_density *density, const int region_size);
void spec_inject_particles(t_particle_vector *part_vector, const int range[][2], const int ppc[2],
		const t_density *part_density, const t_part_data dx[2], const int n_move,
		const t_part_data ufl[3], const t_part_data uth[3]);
void spec_delete(t_species *spec);
void spec_organize_in_tiles(t_species *spec, const int limits_y[2]);

// Utilities
void part_vector_alloc(t_particle_vector *vector, const int size_max);
void part_vector_free(t_particle_vector *vector);
void part_vector_realloc(t_particle_vector *vector, const int new_size);
void part_vector_assign_part(const t_particle_vector *source, const int source_idx,
									t_particle_vector *target, const int target_idx);
void part_vector_memcpy(const t_particle_vector *source, t_particle_vector *target, const int begin,
						 const int size);
void part_vector_mem_advise(t_particle_vector *vector, const int advise, const int device);

// OpenAcc Tasks
void spec_advance_openacc(t_species *restrict const spec, const t_emf *restrict const emf,
		t_current *restrict const current, const int limits_y[2], const int device);
void spec_move_window_openacc(t_species *restrict spec, const int limits_y[2], const int device);
void spec_check_boundaries_openacc(t_species *restrict spec, const int limits_y[2], const int device);
void spec_sort_openacc(t_species *spec, const int limits_y[2], const int device);

#ifdef ENABLE_PREFETCH
void spec_prefetch_openacc(t_particle_vector *part, const int device, void *stream);
#endif

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
void spec_deposit_charge(const t_species *spec, t_part_data *charge);
void spec_rep_charge(t_part_data *restrict charge, const int true_nx[2], const t_fld box[2],
		const int iter_num, const float dt, const bool moving_window, const char path[128]);

// Energy
void spec_calculate_energy(t_species *spec);

#endif
