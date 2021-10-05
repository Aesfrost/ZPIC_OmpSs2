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
#include <GASPI.h>

#include "utilities.h"
#include "zpic.h"

enum smooth_type {
	NONE, BINOMIAL, COMPENSATED
};

typedef struct {
	enum smooth_type xtype, ytype;
	int xlevel, ylevel;
} t_smooth;

typedef struct {

	MPI_Request requests[2 * NUM_ADJ_GRID];

	t_vfld *J;
	t_vfld *J_buf;

	// GASPI segments
	t_vfld *send_J[NUM_ADJ_GRID];
	t_vfld *receive_J[NUM_ADJ_GRID];
	int gaspi_segm_offset_recv[NUM_ADJ_GRID];
	int gaspi_segm_offset_send[NUM_ADJ_GRID];
	int gaspi_remote_offset_send[NUM_ADJ_GRID];

	int gaspi_notif[NUM_ADJ_GRID];

	bool first_comm;
	bool on_right_edge;
	bool on_left_edge;

	// Grid parameters
	int nx[2];
	int nrow;
	int ncol;
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

} t_current;

// Setup
void current_new(t_current *current, int nx[], t_fld box[], float dt, bool on_right_edge, bool on_left_edge);
void current_delete(t_current *current);
void current_link_adj_regions(t_current *current, t_current *current_down, t_current *current_up,
                              t_vfld *gaspi_segm_J, const int segm_offset[2 * NUM_ADJ_GRID],
                              const gaspi_rank_t adj_ranks[NUM_ADJ_GRID], const int region_id,
                              const int region_limits[2][2], const int proc_limits[2][2]);
void current_comm_wait(t_current *current);

// Report ZDF
void current_reconstruct_global_buffer(t_current *current, float *global_buffer, const int offset_y,
                                     const int offset_x, const int sim_nrow, const int jc);
void current_report(const float *restrict global_buffer, const int iter_num, const int true_nx[2],
		const float box[2], const float dt, const char jc, const char path[128]);

// CPU Tasks
void current_wait_comm_x(t_current *current, const int region_id, const int notif_mod);
void current_wait_comm_y(t_current *current, const int notif_mod);

#pragma oss task label("Current Reset") \
	out(current->J_buf[0; current->total_size])
void current_zero(t_current *current);

#pragma oss task label("Current Send X") \
	in(current->J_buf[0; current->total_size]) \
	onready(current_wait_comm_x(current, region_id, NOTIF_ID_CURRENT_ACK))
void current_send_gc_x(t_current *current, const int region_id,
                       const gaspi_rank_t adj_ranks[NUM_ADJ_GRID]);

#pragma oss task label("Current Reduction X") \
	inout(current->J_buf[0; current->total_size]) \
	onready(current_wait_comm_x(current, region_id, NOTIF_ID_CURRENT))
void current_reduction_x(t_current *current, const int region_id,
                         gaspi_rank_t adj_ranks[NUM_ADJ_GRID]);

#pragma oss task label("Current Update GC X") \
	inout(current->J_buf[0; current->total_size]) \
	onready(current_wait_comm_x(current, region_id, NOTIF_ID_CURRENT))
void current_update_gc_x(t_current *current, const int region_id,
                         gaspi_rank_t adj_ranks[NUM_ADJ_GRID]);

#pragma oss task label("Current Filter X") \
	inout(current->J_buf[0; current->total_size])
void current_smooth_x(t_current *current, enum smooth_type type);

#pragma oss task label("Current Send Y") \
	inout(current->J_buf[0; current->total_size]) \
	onready(current_wait_comm_y(current, NOTIF_ID_CURRENT_ACK))
void current_send_gc_y(t_current *current, const int region_id,
                       const gaspi_rank_t adj_ranks[NUM_ADJ_GRID]);

#pragma oss task label("Current Reduction Y") \
	inout(current->J_buf[0; current->overlap_size]) \
	inout(current->receive_J[GRID_DOWN][0; current->overlap_size]) \
	onready(current_wait_comm_y(current, NOTIF_ID_CURRENT))
void current_reduction_y(t_current *current, const int region_id,
                         const gaspi_rank_t adj_ranks[NUM_ADJ_GRID]);

#pragma oss task label("Current Filter Y") \
	inout(current->J_buf[0; current->total_size])
void current_smooth_y(t_current *current, enum smooth_type type);

#endif
