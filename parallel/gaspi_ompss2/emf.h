/*********************************************************************************************
 ZPIC
 emf.h

 Created by Ricardo Fonseca on 10/8/10.
 Modified by Nicolas Guidotti on 11/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#ifndef __EMF__
#define __EMF__

#include "zpic.h"

#include "current.h"
#include "utilities.h"

enum emf_field_type {
	EFLD, BFLD
};

enum emf_laser_type {
	PLANE, GAUSSIAN
};

typedef struct {

	enum emf_laser_type type;		// Laser pulse type

	float start;	// Front edge of the laser pulse, in simulation units
	float fwhm;		// FWHM of the laser pulse duration, in simulation units
	float rise, flat, fall;    // Rise, flat and fall time of the laser pulse, in simulation units

	float a0;		// Normalized peak vector potential of the pulse
	float omega0;    // Laser frequency, normalized to the plasma frequency

	float polarization;

	float W0;		// Gaussian beam waist, in simulation units
	float focus;	// Focal plane position, in simulation units
	float axis;     // Position of optical axis, in simulation units

} t_emf_laser;

typedef struct {

	t_vfld *E;
	t_vfld *B;
	t_vfld *E_buf;
	t_vfld *B_buf;

	// GASPI segments
	t_vfld *send_E[NUM_ADJ_GRID];
	t_vfld *receive_E[NUM_ADJ_GRID];
	t_vfld *send_B[NUM_ADJ_GRID];
	t_vfld *receive_B[NUM_ADJ_GRID];

	int gaspi_segm_offset_recv[NUM_ADJ_GRID];
	int gaspi_segm_offset_send[NUM_ADJ_GRID];
	int gaspi_remote_offset_send[NUM_ADJ_GRID];

	bool on_right_edge;
	bool on_left_edge;

	// Simulation box info
	int nx[2];
	int nrow;
	int gc[2][2];
	t_fld box[2];
	t_fld dx[2];

	int total_size;   // Total size of the buffer
	int overlap_size;   // Size of the overlap

	// Time step
	float dt;

	// Iteration number
	int iter;

	// Moving window
	bool moving_window;
	int n_move;
	bool shift_window_iter;
} t_emf;

// Setup
void emf_new(t_emf *emf, int nx[], t_fld box[], const float dt, const bool on_right_edge, const bool on_left_edge);
void emf_delete(t_emf *emf);
void emf_link_adj_regions(t_emf *emf, t_emf *emf_down, t_emf *emf_up, t_vfld *gaspi_segm_E,
                          t_vfld *gaspi_segm_B, const int segm_offset[2 * NUM_ADJ_GRID],
                          const gaspi_rank_t adj_ranks[NUM_ADJ_GRID], const int region_id,
                          const int region_limits[2][2], const int proc_limits[2][2]);
void emf_add_remote_offset(t_emf *emf, const int region_id, const bool first_region,
                           const bool last_region);
void emf_add_laser(t_emf_laser *laser, t_vfld *restrict E, t_vfld *restrict B, const int nx[2],
		const int nrow, const float dx[2], const int gc[2][2]);

// General Report
double emf_time(void);
double emf_get_energy(t_emf *emf);

// ZDF Report
void emf_reconstruct_global_buffer(const t_emf *emf, float *global_buffer, const int offset_y,
                                   const int offset_x, const int sim_nrow, const char field,
                                   const char fc);
void emf_report(const float *restrict global_buffer, const float box[2], const int true_nx[2],
                const int iter, const float dt, const char field, const char fc, const char path[128]);

// CPU Tasks
#pragma oss task  label("EMF Advance") \
	in(current->J_buf[0; current->total_size]) \
	inout(emf->E_buf[0; emf->total_size]) \
	inout(emf->B_buf[0; emf->total_size])
void emf_advance(t_emf *emf, const t_current *current);

#pragma oss task  label("EMF Update GC X") \
	inout(emf->E_buf[0; emf->total_size]) \
	inout(emf->B_buf[0; emf->total_size])
void emf_update_gc_x(t_emf *emf, const int region_id, const gaspi_rank_t adj_ranks[4]);

#pragma oss task  label("EMF Send X") \
	inout(emf->E_buf[0; emf->total_size]) \
	inout(emf->B_buf[0; emf->total_size])
void emf_send_gc_x(t_emf *emf, const int region_id, const gaspi_rank_t adj_ranks[NUM_ADJ_GRID]);

#pragma oss task  label("EMF Update GC Y") \
	inout(emf->receive_E[GRID_DOWN][0; emf->gc[1][0] * emf->nrow]) \
	inout(emf->receive_B[GRID_DOWN][0; emf->gc[1][0] * emf->nrow]) \
	inout(emf->receive_E[GRID_UP][emf->nrow; emf->gc[1][1] * emf->nrow]) \
	inout(emf->receive_B[GRID_UP][emf->nrow; emf->gc[1][1] * emf->nrow]) \
	inout(emf->E_buf[0; emf->gc[1][0] * emf->nrow]) \
	inout(emf->B_buf[0; emf->gc[1][0] * emf->nrow]) \
	inout(emf->E[emf->nx[1] * emf->nrow; emf->gc[1][1] * emf->nrow]) \
	inout(emf->B[emf->nx[1] * emf->nrow; emf->gc[1][1] * emf->nrow])
void emf_update_gc_y(t_emf *emf, const int region_id, const gaspi_rank_t adj_ranks[4]);

#pragma oss task  label("EMF Send Y") \
	inout(emf->E_buf[0; emf->total_size]) \
	inout(emf->B_buf[0; emf->total_size])
void emf_send_gc_y(t_emf *emf, const int region_id, const gaspi_rank_t adj_ranks[NUM_ADJ_GRID]);

void emf_update_gc_serial(t_vfld *restrict E, t_vfld *restrict B, const int nx[2], const int nrow,
		const int gc[2][2]);
#endif
