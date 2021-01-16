/*********************************************************************************************
 ZPIC
 current.h

 Created by Ricardo Fonseca on 12/8/10.
 Modified by Nicolas Guidotti on 14/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/


#ifndef __CURRENT__
#define __CURRENT__

#include <stdbool.h>
#include <stddef.h>
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
	int overlap_size;

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

// ZDF report
void current_reconstruct_global_buffer(t_current *current, float *global_buffer, const int offset,
		const int jc);
void current_report(const float *restrict global_buffer, const int iter_num, const int true_nx[2],
		const float box[2], const float dt, const char jc, const char path[128]);

// CPU Tasks
#pragma oss task out(current->J_buf[0; current->total_size]) label("Current Reset")
void current_zero(t_current *current);

#pragma oss task inout(current->J_buf[0; current->overlap_size]) \
inout(current->J_upper[-current->gc[0][0]; current->overlap_size]) \
label("Current Reduction Y")
void current_reduction_y(t_current *current); // Each region only update the zone in the top edge

#pragma oss task inout(current->J_buf[0; current->total_size]) label("Current Reduction X")
void current_reduction_x(t_current *current);

#pragma oss task inout(current->J_buf[0; current->overlap_size]) \
inout(current->J_upper[-current->gc[0][0]; current->overlap_size]) \
label("Current Update GC")
void current_gc_update_y(t_current *current);

#pragma oss task inout(current->J_buf[0; current->total_size]) \
label("Current Smooth X")
void current_smooth_x(t_current *current);

#pragma oss task inout(current->J_buf[0; current->total_size]) label("Current Smooth Y")
void current_smooth_y(t_current *current, enum smooth_type type);

// OpenAcc Tasks
#pragma oss task out(current->J_buf[0; current->total_size]) \
	label("Current Reset (GPU)")
void current_zero_openacc(t_current *current, const int device);

#pragma oss task inout(current->J_buf[0; current->overlap_size]) \
inout(current->J_upper[-current->gc[0][0]; current->overlap_size]) \
label("Current Reduction Y (GPU)")
void current_reduction_y_openacc(t_current *current, const int device);

#pragma oss task inout(current->J_buf[0; current->total_size]) \
label("Current Reduction X (GPU)")
void current_reduction_x_openacc(t_current *current, const int device);

#pragma oss task inout(current->J_buf[0; current->total_size]) \
label("Current Smooth X (GPU)")
void current_smooth_x_openacc(t_current *current, const int device);

#pragma oss task inout(current->J_buf[0; current->overlap_size]) \
inout(current->J_upper[-current->gc[0][0]; current->overlap_size]) \
label("Current Update GC (GPU)")
void current_gc_update_y_openacc(t_current *current, const int device);

// Prefetch
#ifdef ENABLE_PREFETCH
void current_prefetch_openacc(t_vfld *buf, const size_t size, const int device);
#endif

#endif