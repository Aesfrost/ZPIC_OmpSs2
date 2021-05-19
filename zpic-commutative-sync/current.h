/*********************************************************************************************
 ZPIC
 current.h

 Created by Ricardo Fonseca on 12/8/10.
 Modified by Nicolas Guidotti on 11/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#ifndef __CURRENT__
#define __CURRENT__

#include <stdbool.h>

#include "zpic.h"

enum smooth_type {
	NONE, BINOMIAL, COMPENSATED
};

typedef struct {
	enum smooth_type xtype, ytype;
	int xlevel, ylevel;
} t_smooth;

typedef struct {

	t_vfld *J;

	t_vfld *J_buf;

	// Grid parameters
	int nx[2];
	int nrow;
	int gc[2][2];
	int total_size;
	int overlap_zone;

	// Box size
	t_fld box[2];

	// Cell size
	t_fld dx[2];

	// Current smoothing
	t_smooth smooth;

	// Time step
	float dt;

	// Iteration number
	int iter;

	// Moving window
	bool moving_window;

	// Pointer to the overlap zone (in the current buffer) in the region above
	t_vfld *J_upper;

} t_current;

// Setup
void current_new(t_current *current, int nx[], t_fld box[], float dt);
void current_delete(t_current *current);
void current_overlap_zone(t_current *current, t_current *upper_current);

// Report ZDF
void current_reconstruct_global_buffer(t_current *current, float *global_buffer, const int offset,
		const int jc);
void current_report(const float *restrict global_buffer, const int iter_num, const int true_nx[2],
		const float box[2], const float dt, const char jc, const char path[128]);

// CPU Tasks
void current_zero(t_current *current);

#pragma oss task inout(current->J_buf[0; current->overlap_zone]) \
inout(current->J_upper[-current->gc[0][0]; current->overlap_zone]) \
label("Current Reduction Y")
void current_reduction_y(t_current *current); // Each region only update the zone in the top edge

#pragma oss task inout(current->J[begin * current->nrow - current->gc[0][0]; end * current->nrow]) label("Current Reduction X")
void current_reduction_x(t_current *current, const int begin, const int end);

#pragma oss task inout(current->J_buf[0; current->overlap_zone]) \
inout(current->J_upper[-current->gc[0][0]; current->overlap_zone]) \
label("Current Update GC")
void current_gc_update_y(t_current *current); // Each region only update the zone in the top edge

#pragma oss task inout(current->J[-current->gc[0][0]; current->nx[1] * current->nrow]) label("Current Smooth X")
void current_smooth_x(t_current *current);

#pragma oss task inout(current->J_buf[0; current->total_size]) label("Current Smooth Y")
void current_smooth_y(t_current *current, enum smooth_type type);

#endif
