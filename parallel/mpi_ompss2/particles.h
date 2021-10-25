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
#include <mpi.h>

#include "zpic.h"
#include "emf.h"
#include "current.h"

#define MAX_SPNAME_LEN 32
#define COMM_NPC_FACTOR 20

#define LTRIM(x) (x >= 1.0f) - (x < 0.0f)

typedef struct {
	int ix, iy;
	t_part_data x, y;
	t_part_data ux, uy, uz;

	// Mark the particle as invalid (the particle exited the region)
	bool invalid;

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
} t_part_vector;

typedef struct {
	char name[MAX_SPNAME_LEN];

	// Particle data buffer
	t_part_vector main_vector;
	t_part_vector incoming_part[NUM_ADJ_PART];    // Temporary buffer for incoming particles
	t_part_vector *outgoing_part[NUM_ADJ_PART];

	bool inter_proc_comm[NUM_ADJ_PART];

	int num_requests_np;
	int num_requests_part;
	MPI_Request mpi_requests_np[2 * NUM_ADJ_PART];
	MPI_Request mpi_requests_part[2 * NUM_ADJ_PART];

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

	// Simulation info
	t_part_data dx[2];

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
void spec_inject_particles(t_part_vector *part_vector, const int range[][2],
                           const int proc_limits[2][2], const int ppc[2],
                           const t_density *part_density, const t_part_data dx[2], const int n_move,
                           const t_part_data ufl[3], const t_part_data uth[3]);
void spec_create_incoming_buffers(t_species *spec, const int region_nx[2], const bool first_region,
                                  const bool last_region);
void spec_link_adj_regions(t_species *spec, t_part_vector *adj_spec[8], const int region_nx[2]);
void spec_delete(t_species *spec);

// Report - General
double spec_time(void);
double spec_perf(void);

// CPU Tasks
#pragma oss task label("Spec Advance") \
		in(emf->E_buf[0; emf->total_size]) \
		in(emf->B_buf[0; emf->total_size]) \
		inout(spec->main_vector) \
		out(*spec->outgoing_part[PART_UP]) \
		out(*spec->outgoing_part[PART_UP_LEFT]) \
		out(*spec->outgoing_part[PART_UP_RIGHT]) \
		out(*spec->outgoing_part[PART_DOWN]) \
		out(*spec->outgoing_part[PART_DOWN_LEFT]) \
		out(*spec->outgoing_part[PART_DOWN_RIGHT]) \
		inout(current->J_buf[0; current->total_size])
void spec_advance(t_species *spec, const t_emf *emf, t_current *current,
                  const int region_limits[2][2], const int sim_nx[2]);

#pragma oss task label("Spec Send NP") \
		inout(spec->main_vector) \
		in(spec->incoming_part[PART_DOWN_LEFT]) \
		in(spec->incoming_part[PART_DOWN_RIGHT]) \
		in(spec->incoming_part[PART_UP_LEFT]) \
		in(spec->incoming_part[PART_UP_RIGHT])
void spec_send_outgoing_np(t_species *spec, const int region_id, const int spec_id,
                           unsigned int adj_ranks[NUM_ADJ_PART]);

#pragma oss task label("Spec Send Particles") \
		inout(spec->main_vector) \
		in(spec->incoming_part[PART_DOWN_LEFT]) \
		in(spec->incoming_part[PART_DOWN_RIGHT]) \
		in(spec->incoming_part[PART_UP_LEFT]) \
		in(spec->incoming_part[PART_UP_RIGHT])
void spec_send_particles(t_species *spec, const int region_id, const int spec_id,
                         unsigned int adj_ranks[NUM_ADJ_PART]);

#pragma oss task label("Spec Receive Particles") \
		inout(spec->main_vector) \
		in(spec->incoming_part[PART_DOWN_LEFT]) \
		in(spec->incoming_part[PART_DOWN_RIGHT]) \
		in(spec->incoming_part[PART_UP_LEFT]) \
		in(spec->incoming_part[PART_UP_RIGHT])
void spec_receive_particles(t_species *spec);

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
                  const float pha_range[][2], const int iter_num, const float dt,
                  const char path[128]);

// Charge map
void spec_deposit_charge(const t_species *spec, t_part_data *charge, const int nrow);
void spec_rep_charge(t_part_data *restrict charge, const int true_nx[2], const t_fld box[2],
		const int iter_num, const float dt, const bool moving_window, const char path[128]);

// Energy
void spec_calculate_energy(t_species *spec);

#endif
