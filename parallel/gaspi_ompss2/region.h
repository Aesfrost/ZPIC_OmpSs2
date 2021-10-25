/*********************************************************************************************
 ZPIC
 region.h

 Created by Nicolas Guidotti on 11/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#ifndef REGION
#define REGION

#include <stdbool.h>
#include "particles.h"
#include "emf.h"
#include "current.h"

// The regions are stored in a double linked list
typedef struct Region {
	int id;

	double iter_time;

	int nx[2];   			// Region size
	int limits[2][2];   	// Limits of the region

	struct Region *next;   // Pointer to the uper region (j => j_max)
	struct Region *prev;   // Pointer to the bottom region (j < j_min)

	// Local species
	int n_species;
	t_species *species;

	// Local current and emf
	t_current local_current;
	t_emf local_emf;

} t_region;

void region_new(t_region *region, int id, int n_regions, int proc_nx[2], int proc_limits[2][2],
                float proc_box[], int n_spec, t_species *spec, float dt, bool on_right_edge,
                bool on_left_edge, t_region *prev_region, t_region *next_region);
void region_link_adj_part(t_region *region, t_part **gaspi_segm_part,
                          const int **gaspi_segm_offset, const int proc_limits[2][2],
                          const int sim_nx[2]);
void region_link_adj_grid(t_region *region, t_vfld *gaspi_segm_J, t_vfld *gaspi_segm_E,
							t_vfld *gaspi_segm_B, const int emf_segm_offset[8],
							const int current_segm_offset[8], const gaspi_rank_t adj_ranks[4],
							const int proc_limits[2][2], const int sim_nx[2]);
void region_set_moving_window(t_region *region);
void region_delete(t_region *region);

// Report
void region_charge_report(const t_region *region, t_part_data *charge, int i_spec);
void region_emf_report(const t_region *region, t_fld *restrict E_mag, t_fld *restrict B_mag, const int nrow);
void region_report_iter_time(const t_region *region, double time[], int i);

#endif
