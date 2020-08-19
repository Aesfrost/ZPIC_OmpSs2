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

enum CURRENT_SMOOTH_MODE {
	SMOOTH_X, BINOMIAL_Y, COMPENSATED_Y, CURRENT_UPDATE_GC
};

enum EMF_UPDATE {
	EMF_ADVANCE, EMF_UPDATE_GC
};

// The GPU regions are setup a little bit different than a normal (CPU) regions. First, we define the percentage of regions that
// will be dedicated to the GPU (e.g., 50% of the all the regions), obtaining the total size of the GPU block.
// Then, we split this block by the number of GPU regions (as being defined previously).
// For example, if GPU percentage = 50%, Number of GPU regions = 2 and Number of regions in total = 64,
// there will be 32 CPU regions and 2 GPU regions, but each GPU have a size equal to 16 CPU regions.

// The regions are stored in a double linked list
typedef struct Region
{
	int id;
	bool enable_gpu; // Mark the region as a GPU region (all the computational steps will be done using OpenAcc)
	double iter_time;

	int iter;

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
		float box[], float dt, float gpu_percentage, int n_gpu_regions, t_region *prev_region);
void region_init(t_region *region);
void region_set_moving_window(t_region *region);
void region_add_laser(t_region *region, t_emf_laser *laser);
void region_delete(t_region *region);

void region_advance(t_region *region); // Recursive call to other regions

void region_charge_report(const t_region *region, t_part_data *charge, int i_spec);
void region_emf_report(const t_region *region, t_fld *restrict E_mag, t_fld *restrict B_mag, const int nrow);
void region_report_iter_time(const t_region *region, double time[], int i);

int get_n_regions();
int get_gpu_regions_effective();

#endif
