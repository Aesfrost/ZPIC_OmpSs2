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
typedef struct Region
{
	int id;

	double iter_time;

	int nx[2]; // Region size
	int limits_y[2]; // Limits of the region in y

	struct Region *next; // Pointer to the bottom region (j => j_max)
	struct Region *prev; // Pointer to the upper region (j < j_min)

	// Local species
	int n_species;
	t_species *species;

	// Local current and emf
	t_current local_current;
	t_emf local_emf;

} t_region;

void region_new(t_region *region, int n_regions, int nx[2], int id, int n_spec, t_species *spec,
		float box[], float dt, t_region *prev_region, t_region *next_region);
void region_init(t_region *region);
void region_set_moving_window(t_region *region);
void region_add_laser(t_region *region, t_emf_laser *laser);
void region_delete(t_region *region);

void region_advance(t_region *regions, const int n_regions);

void region_charge_report(const t_region *region, t_part_data *charge, int i_spec);
void region_emf_report(const t_region *region, t_fld *restrict E_mag, t_fld *restrict B_mag, const int nrow);
void region_report_iter_time(const t_region *region, double time[], int i);

#endif
