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

enum emf_diag {
	EFLD, BFLD
};

typedef struct {

	t_vfld *E;
	t_vfld *B;

	t_vfld *E_buf;
	t_vfld *B_buf;

	// Simulation box info
	int nx[2];
	int nrow;
	int gc[2][2];
	t_fld box[2];
	t_fld dx[2];

	int total_size; // Total size of the buffer
	int overlap; // Size of the overlap

	// Time step
	float dt;

	// Iteration number
	int iter;

	// Moving window
	bool moving_window;
	int n_move;

	// Pointer to the overlap zone (in the E/B buffer) in the region above
	t_vfld *B_upper, *E_upper;

} t_emf;

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

// Setup
void emf_new(t_emf *emf, int nx[], t_fld box[], const float dt);
void emf_delete(t_emf *emf);
void emf_overlap_zone(t_emf *emf, t_emf *upper);
void emf_add_laser(t_emf *const emf, t_emf_laser *laser, int offset_y);
void div_corr_x(t_emf *emf);

// General Report
double emf_time(void);
double emf_get_energy(t_emf *emf);

// ZDF Report
void emf_reconstruct_global_buffer(const t_emf *emf, float *global_buffer, const int offset,
		const char field, const char fc);
void emf_report(const float *restrict global_buffer, const float box[2], const int true_nx[2],
		const int iter, const float dt, const char field, const char fc, const char path[128]);

// CSV Report
void emf_report_magnitude(const t_emf *emf, t_fld *restrict E_mag,
		t_fld *restrict B_mag, const int nrow, const int offset);

// CPU Tasks
#pragma oss task in(current->J_buf[0; current->total_size]) \
inout(emf->E_buf[0; emf->total_size]) \
inout(emf->B_buf[0; emf->total_size]) \
label(EMF Advance)
void emf_advance(t_emf *emf, const t_current *current);

#pragma oss task inout(emf->B_buf[0; emf->overlap]) \
inout(emf->B_upper[-emf->gc[0][0]; emf->overlap]) \
inout(emf->E_buf[0; emf->overlap]) \
inout(emf->E_upper[-emf->gc[0][0]; emf->overlap]) \
label(EMF Update GC)
void emf_update_gc_y(t_emf *emf); // Each region is update the ghost cells in the top edge

void emf_update_gc_x(t_emf *emf);

#endif
