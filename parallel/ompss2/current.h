/*
 *  current.h
 *  zpic
 *
 *  Created by Ricardo Fonseca on 12/8/10.
 *  Copyright 2010 Centro de Física dos Plasmas. All rights reserved.
 *
 */

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

void current_new(t_current *current, int nx[], t_fld box[], float dt);
void current_delete(t_current *current);
void current_zero(t_current *current);
void current_overlap_zone(t_current *current, t_current *upper_current);
//void current_update(t_current *current);

void current_reduction_y(t_current *current); // Each region only update the zone in the top edge
void current_reduction_x(t_current *current);
void current_gc_update_y(t_current *current);
void current_smooth_x(t_current *current);
//void current_smooth_y(t_current *current);

void current_report(const t_current *current, const char jc);
void current_smooth(t_current *const current);

#endif
