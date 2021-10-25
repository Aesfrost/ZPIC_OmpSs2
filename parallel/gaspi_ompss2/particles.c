/*********************************************************************************************
 ZPIC
 particles.c

 Created by Ricardo Fonseca on 11/8/10.
 Modified by Nicolas Guidotti on 11/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#include "particles.h"

#include "random.h"
#include "emf.h"
#include "current.h"
#include "utilities.h"
#include "zdf.h"
#include "timer.h"
#include "task_management.h"

/*********************************************************************************************
 Initialization
 *********************************************************************************************/

// Inject the particles in the simulation
void spec_inject_particles(t_part_vector *part_vector, const int range[][2],
                           const int proc_limits[2][2], const int ppc[2],
                           const t_density *part_density, const t_part_data dx[2], const int n_move,
                           const t_part_data ufl[3], const t_part_data uth[3])
{
	int start_part = part_vector->size;
	int start_x, end_x;

	// Calculate particle positions inside the cell
	const int npc = ppc[0] * ppc[1];
	t_part_data const dpcx = 1.0f / ppc[0];
	t_part_data const dpcy = 1.0f / ppc[1];

	float *poscell = malloc(2 * npc * sizeof(t_part_data));

	int ip = 0;
	for (int j = 0; j < ppc[1]; j++)
	{
		for (int i = 0; i < ppc[0]; i++)
		{
			poscell[ip] = dpcx * (i + 0.5);
			poscell[ip + 1] = dpcy * (j + 0.5);
			ip += 2;
		}
	}

	// Set position of particles in the specified grid range according to the density profile
	switch (part_density->type)
	{
		case STEP:    // Step like density profile

			// Get edge position normalized to cell size;
			start_x = part_density->start / dx[0] - n_move;
			if (range[0][0] > start_x) start_x = range[0][0];

			end_x = range[0][1];
			break;

		case SLAB:    // Slab like density profile
			// Get edge position normalized to cell size;
			start_x = part_density->start / dx[0] - n_move;
			end_x = part_density->end / dx[0] - n_move;

			if (start_x < range[0][0]) start_x = range[0][0];
			if (end_x > range[0][1]) end_x = range[0][1];
			break;

		default:    // Uniform density
			start_x = range[0][0];
			end_x = range[0][1];

	}

	// Get maximum number of particles to inject
	int np_inj = (MIN_VALUE(end_x, proc_limits[0][1]) - MAX_VALUE(start_x, proc_limits[0][0]))
				* (MIN_VALUE(range[1][1], proc_limits[1][1]) - MAX_VALUE(range[1][0], proc_limits[1][0]))
				* ppc[0] * ppc[1];

	if (np_inj > 0)
	{
		// Check if buffer is large enough and if not reallocate
		if (start_part + np_inj > part_vector->size_max)
		{
			part_vector->size_max = ((part_vector->size_max + np_inj) / 1024 + 1) * 1024;

			if(!part_vector->data) part_vector->data = malloc(part_vector->size_max * sizeof(t_part));
			else realloc_vector(&part_vector->data, part_vector->size, part_vector->size_max, sizeof(t_part));
		}

		ip = start_part;

#ifndef TEST

		for (int j = range[1][0]; j < range[1][1]; j++)
		{
			for (int i = start_x; i < end_x; i++)
			{
				// Init particles only inside the process
				if (j < proc_limits[1][1] && j >= proc_limits[1][0])
				{
					if (i < proc_limits[0][1] && i >= proc_limits[0][0])
					{
						for (int k = 0; k < npc; k++)
						{
							part_vector->data[ip].ix = i;
							part_vector->data[ip].iy = j;
							part_vector->data[ip].x = poscell[2 * k];
							part_vector->data[ip].y = poscell[2 * k + 1];
							part_vector->data[ip].ux = ufl[0] + uth[0] * rand_norm();
							part_vector->data[ip].uy = ufl[1] + uth[1] * rand_norm();
							part_vector->data[ip].uz = ufl[2] + uth[2] * rand_norm();
							part_vector->data[ip].invalid = false;
							ip++;
						}
					} else
					{
						// Advance RNG list
						for (int k = 0; k < npc; k++)
						{
							rand_norm();
							rand_norm();
							rand_norm();
						}
					}
				} else
				{
					// Advance RNG list
					for (int k = 0; k < npc; k++)
					{
						rand_norm();
						rand_norm();
						rand_norm();
					}
				}
			}
		}

#else
		end_x = MIN_VALUE(end_x, proc_limits[0][1]);
		start_x = MAX_VALUE(start_x, proc_limits[0][0]);

		int start_y = MAX_VALUE(range[1][0], proc_limits[1][0]);
		int end_y = MIN_VALUE(range[1][1], proc_limits[1][1]);

		for (int j = start_y; j < end_y; j++)
		{
			for (int i = start_x; i < end_x; i++)
			{
				for (int k = 0; k < npc; k++)
				{
					part_vector->data[ip].ix = i;
					part_vector->data[ip].iy = j;
					part_vector->data[ip].x = poscell[2 * k];
					part_vector->data[ip].y = poscell[2 * k + 1];
					part_vector->data[ip].ux = ufl[0] + uth[0] * rand_norm();
					part_vector->data[ip].uy = ufl[1] + uth[1] * rand_norm();
					part_vector->data[ip].uz = ufl[2] + uth[2] * rand_norm();
					part_vector->data[ip].invalid = false;
					ip++;
				}
			}
		}

#endif

		part_vector->size = ip;
	}

	free(poscell);
}

// Constructor
void spec_new(t_species *spec, char name[], const t_part_data m_q, const int ppc[],
              const t_part_data *ufl, const t_part_data *uth, const int nx[], t_part_data box[],
              const float dt, t_density *density)
{
	// Species name
	strncpy(spec->name, name, MAX_SPNAME_LEN);

	int npc = 1;
	// Store species data
	for (int i = 0; i < 2; i++)
	{
		spec->ppc[i] = ppc[i];
		npc *= ppc[i];

		spec->dx[i] = box[i] / nx[i];
	}

	spec->m_q = m_q;
	spec->q = copysign(1.0f, m_q) / npc;

	spec->dt = dt;
	spec->energy = 0;

	// Initialize particle buffer
	spec->main_vector.size_max = 0;
	spec->main_vector.data = NULL;
	spec->main_vector.size = 0;

	// The left and right regions are always in another process
	for (int i = 0; i < 8; ++i)
	{
		spec->incoming_part[i].data = NULL;
		spec->incoming_part[i].size = 0;
		spec->incoming_part[i].size_max = 0;

		spec->gaspi_segm_offset_send[i] = -1;
		spec->gaspi_segm_offset_recv[i] = -1;
	}

	// Initialize density profile
	if (density)
	{
		spec->density = *density;
		if (spec->density.n == 0.) spec->density.n = 1.0;
	} else
	{
		// Default values
		spec->density = (t_density) {.type = UNIFORM, .n = 1.0};
	}

	// Initialize temperature profile
	if (ufl)
	{
		for (int i = 0; i < 3; i++)
			spec->ufl[i] = ufl[i];
	} else
	{
		for (int i = 0; i < 3; i++)
			spec->ufl[i] = 0;
	}

	// Density multiplier
	spec->q *= fabsf(spec->density.n);

	if (uth)
	{
		for (int i = 0; i < 3; i++)
			spec->uth[i] = uth[i];
	} else
	{
		for (int i = 0; i < 3; i++)
			spec->uth[i] = 0;
	}

	// Reset iteration number
	spec->iter = 0;

	// Reset moving window information
	spec->moving_window = false;
	spec->n_move = 0;
}

void spec_delete(t_species *spec)
{
	free(spec->main_vector.data);
	spec->main_vector.size = -1;

	for (int i = 0; i < NUM_ADJ_PART; i++)
	{
		if (spec->gaspi_segm_offset_send[i] == -1)
		{
			free(spec->incoming_part[i].data);
			spec->incoming_part[i].size = -1;
		}

		if (spec->gaspi_segm_offset_send[i] >= 0)
			free(spec->outgoing_part[i]);
	}
}

/*********************************************************************************************
 Communication
 *********************************************************************************************/

void spec_create_incoming_buffers(t_species *spec, t_part *part_gaspi_segm,
                                  const int *gaspi_segm_offset, const int spec_id,
                                  const int region_id, const int region_nx[2],
                                  const int region_limits[2][2], const int proc_limits[2][2],
                                  gaspi_rank_t adj_ranks[8], const bool first_region,
                                  const bool last_region)
{
	const int npc = COMM_NPC_FACTOR * spec->ppc[0] * spec->ppc[1];

							//  Left				Centre		Right
	const int size_per_dir[] = {1, 				region_nx[0], 	1,				// Down
	                            region_nx[1], 					region_nx[1],	// Centre
	                            1, 				region_nx[0], 	1};				// Up

	bool inter_process_comm[NUM_ADJ_PART];
	for (int dir = 0; dir < NUM_ADJ_PART; ++dir)
		inter_process_comm[dir] = true;

	if(!first_region)
	{
		inter_process_comm[PART_DOWN_LEFT] = false;
		inter_process_comm[PART_DOWN] = false;
		inter_process_comm[PART_DOWN_RIGHT] = false;
	}

	if(!last_region)
	{
		inter_process_comm[PART_UP_LEFT] = false;
		inter_process_comm[PART_UP] = false;
		inter_process_comm[PART_UP_RIGHT] = false;
	}

	CHECK_GASPI_ERROR(gaspi_wait(DEFAULT_QUEUE, GASPI_BLOCK));

	for (int dir = 0; dir < NUM_ADJ_PART; ++dir)
	{
		if(inter_process_comm[dir])
		{
			int notif_id;
			spec->gaspi_segm_offset_recv[dir] = gaspi_segm_offset[SEGM_OFFSET_PART(dir, GASPI_RECV)];

			if (dir == PART_RIGHT || dir == PART_LEFT)
			{
				spec->gaspi_segm_offset_recv[dir] += (region_limits[1][0] - proc_limits[1][0]) * npc;
				notif_id = NOTIFICATION_ID(OPPOSITE_DIR(dir), region_id, spec_id);
			}else notif_id = NOTIFICATION_ID(OPPOSITE_DIR(dir), 0, spec_id);

			spec->incoming_part[dir].data = part_gaspi_segm + spec->gaspi_segm_offset_recv[dir];
			spec->incoming_part[dir].size = 0;
			spec->incoming_part[dir].size_max = npc * size_per_dir[dir];

			const int value = spec->gaspi_segm_offset_recv[dir] + COMM_PART_PING;
			CHECK_GASPI_ERROR(gaspi_notify(PART_SEGMENT_ID(spec_id), adj_ranks[dir], notif_id, value,
			                               DEFAULT_QUEUE, GASPI_BLOCK));
		}else
		{
			spec->incoming_part[dir].size_max = size_per_dir[dir] * npc;
			spec->incoming_part[dir].data = malloc(spec->incoming_part[dir].size_max * sizeof(t_part));
			spec->incoming_part[dir].size = 0;
			spec->gaspi_segm_offset_recv[dir] = -1;   // Local communication
		}
	}
}

// NULL denote that the adjacent regions is located in another process
void spec_link_adj_regions(t_species *spec, t_part_vector *adj_spec[8], t_part *part_gaspi_segm,
                           const int *gaspi_segm_offset, const int spec_id, const int region_id,
                           const int region_nx[2], const int region_limits[2][2],
                           const int proc_limits[2][2])
{
							 // Left			Centre			Right
	const int size_per_dir[] = {1, 				region_nx[0], 	1,				// Down
	                            region_nx[1], 					region_nx[1],	// Centre
	                            1, 				region_nx[0], 	1};				// Up
	const int ppc = COMM_NPC_FACTOR * spec->ppc[0] * spec->ppc[1];

	for (int i = 0; i < NUM_ADJ_PART; ++i)
	{
		// The adjacent region is on the same process
		if (adj_spec[i])
		{
			spec->outgoing_part[i] = adj_spec[i];
			spec->gaspi_segm_offset_send[i] = -1;   // Local communication

		} else   // The adjacent region is on another process
		{
			gaspi_notification_id_t id;
			gaspi_notification_t value;
			int notif_id;

			// Link the outgoing buffer to the GASPI particle segment and calculate the offset in this segment
			spec->outgoing_part[i] = malloc(sizeof(t_part_vector));
			spec->gaspi_segm_offset_send[i] = gaspi_segm_offset[SEGM_OFFSET_PART(i, GASPI_SEND)];

			// Segment offset based on the region id
			if (i == PART_RIGHT || i == PART_LEFT)
			{
				spec->gaspi_segm_offset_send[i] += (region_limits[1][0] - proc_limits[1][0]) * ppc;
				notif_id = NOTIFICATION_ID(i, region_id, spec_id);
			} else notif_id = NOTIFICATION_ID(i, 0, spec_id);

			spec->outgoing_part[i]->data = part_gaspi_segm + spec->gaspi_segm_offset_send[i];
			spec->outgoing_part[i]->size = 0;
			spec->outgoing_part[i]->size_max = ppc * size_per_dir[i];

			CHECK_GASPI_ERROR(gaspi_notify_waitsome(PART_SEGMENT_ID(spec_id), notif_id, 1, &id, GASPI_BLOCK));
			CHECK_GASPI_ERROR(gaspi_notify_reset(PART_SEGMENT_ID(spec_id), id, &value));
			spec->gaspi_remote_offset_send[i] = value - COMM_PART_PING;
		}
	}
}

// Add the particles in the input buffers to the output_vector
void spec_merge_vectors(t_part_vector *dest, t_part_vector *source)
{
	int i = 0;
	int size = dest->size;
	int size_temp = source->size;

	for (int j = 0; j < size_temp; j++)   // Loop through all elements in the input vector
	{
		if (!source->data[j].invalid)
		{
			// Find "holes" left by an invalid particle in the output vector
			while (i < size && !dest->data[i].invalid) i++;

			if (i < size) dest->data[i] = source->data[j];
			else dest->data[dest->size++] = source->data[j];
		}
	}

	//Clean the temp buffer
	source->size = 0;
}

void spec_send_particles(t_species *spec, const int region_id, const int spec_id,
                         gaspi_rank_t adj_ranks[NUM_ADJ_PART])
{
	const unsigned int queue = get_gaspi_queue(region_id);

	if (spec->gaspi_segm_offset_recv[PART_DOWN_LEFT] < 0)
		spec_merge_vectors(spec->outgoing_part[PART_LEFT],
		                   &spec->incoming_part[PART_DOWN_LEFT]);

	if (spec->gaspi_segm_offset_recv[PART_UP_LEFT] < 0)
		spec_merge_vectors(spec->outgoing_part[PART_LEFT],
		                   &spec->incoming_part[PART_UP_LEFT]);

	if (spec->gaspi_segm_offset_recv[PART_DOWN_RIGHT] < 0)
		spec_merge_vectors(spec->outgoing_part[PART_RIGHT],
		                   &spec->incoming_part[PART_DOWN_RIGHT]);

	if (spec->gaspi_segm_offset_recv[PART_UP_RIGHT] < 0)
		spec_merge_vectors(spec->outgoing_part[PART_RIGHT],
		                   &spec->incoming_part[PART_UP_RIGHT]);

	if(spec->iter > 1)
	{
		int notif_ids[8];

		for (int dir = 0; dir < NUM_ADJ_PART; dir++)
		{
			if (spec->gaspi_segm_offset_send[dir] >= 0)
			{
				if (dir == PART_RIGHT || dir == PART_LEFT)
					notif_ids[dir] = NOTIFICATION_ID(dir, region_id, NOTIF_ID_PART_ACK(spec_id));
				else notif_ids[dir] = NOTIFICATION_ID(dir, 0, NOTIF_ID_PART_ACK(spec_id));
			}else notif_ids[dir] = -1;
		}

		gaspi_recv(PART_SEGMENT_ID(spec_id), notif_ids, COMM_PART_ACK);
	}

	for (int dir = 0; dir < NUM_ADJ_PART; dir++)
	{
		if (spec->outgoing_part[dir]->size > spec->outgoing_part[dir]->size_max)
		{
			gaspi_rank_t rank;
			gaspi_proc_rank(&rank);
			fprintf(stderr, "Process %d: Error - Overflow in outgoing particles buffer (%d)\n",
			        rank, dir);
			fflush(stderr);
			exit(1);
		}

		if (spec->gaspi_segm_offset_send[dir] >= 0)
		{
			const int opposite = OPPOSITE_DIR(dir);

			int id;
			if (dir == PART_RIGHT || dir == PART_LEFT)
				id = NOTIFICATION_ID(opposite, region_id, NOTIF_ID_PART(spec_id));
			else id = NOTIFICATION_ID(opposite, 0, NOTIF_ID_PART(spec_id));

			if(spec->outgoing_part[dir]->size != 0)
			{
				CHECK_GASPI_ERROR(gaspi_write_notify(PART_SEGMENT_ID(spec_id), 		// Local segment ID
						spec->gaspi_segm_offset_send[dir] * sizeof(t_part),			// Local segment offset
						adj_ranks[dir],												// Rank of the receiving process
						PART_SEGMENT_ID(spec_id),									// Remote segment ID
						spec->gaspi_remote_offset_send[dir] * sizeof(t_part),		// Remote segment offset
						spec->outgoing_part[dir]->size * sizeof(t_part),			// Size
						id,															// Notification ID
						spec->outgoing_part[dir]->size + COMM_PART_WRITE,			// Notification value
						queue,														// Queue
						GASPI_BLOCK));												// Timeout in ms

				// Clean outgoing buffer
				spec->outgoing_part[dir]->size = 0;
			}else
			{
				CHECK_GASPI_ERROR(gaspi_notify(PART_SEGMENT_ID(spec_id),
				                               adj_ranks[dir],
				                               id,
				                               COMM_PART_WRITE,
				                               queue,
				                               GASPI_BLOCK));
			}
		}
	}
}

void spec_receive_particles(t_species *spec, const int region_id, const int spec_id,
                            const gaspi_rank_t adj_ranks[8])
{
	int np_inj = 0;
	const unsigned int queue = get_gaspi_queue(region_id);

#ifdef ENABLE_TASKING

	gaspi_notification_t value;
	bool received = true;
	int notif_ids[8];

	for (int i = 0; i < NUM_ADJ_PART; i++)
	{
		if (spec->gaspi_segm_offset_recv[i] >= 0)
		{
			if (i == PART_RIGHT || i == PART_LEFT)
				notif_ids[i] = NOTIFICATION_ID(i, region_id, NOTIF_ID_PART(spec_id));
			else notif_ids[i] = NOTIFICATION_ID(i, 0, NOTIF_ID_PART(spec_id));

			received &= gaspi_notify_test(PART_SEGMENT_ID(spec_id), notif_ids[i]);
		}else notif_ids[i] = -1;
	}

	if(!received)
		block_comm_task(PART_SEGMENT_ID(spec_id), notif_ids);

	for (int i = 0; i < NUM_ADJ_PART; i++)
	{
		if (notif_ids[i] >= 0)
		{
			CHECK_GASPI_ERROR(gaspi_notify_reset(PART_SEGMENT_ID(spec_id), notif_ids[i], &value));
			spec->incoming_part[i].size = value - COMM_PART_WRITE;
		}

		np_inj += spec->incoming_part[i].size;
	}

#else

	for (int i = 0; i < NUM_ADJ_PART; i++)
	{
		if (spec->gaspi_segm_offset_recv[i] >= 0)
		{
			gaspi_notification_t value;
			gaspi_notification_id_t id;

			int notif_id = NOTIFICATION_ID(i, 0, NOTIF_ID_PART(spec_id));
			if (i == PART_RIGHT || i == PART_LEFT)
				notif_id = NOTIFICATION_ID(i, region_id, NOTIF_ID_PART(spec_id));

			CHECK_GASPI_ERROR(gaspi_notify_waitsome(PART_SEGMENT_ID(spec_id),
			                                        notif_id, 1, &id, GASPI_BLOCK));
			CHECK_GASPI_ERROR(gaspi_notify_reset(PART_SEGMENT_ID(spec_id), id, &value));

			spec->incoming_part[i].size = value - COMM_PART_WRITE;
		}

		np_inj += spec->incoming_part[i].size;
	}

#endif

	if (spec->main_vector.size + np_inj > spec->main_vector.size_max)
	{
		spec->main_vector.size_max = ((spec->main_vector.size_max + np_inj) / 1024 + 1) * 1024;
		realloc_vector((void**) &spec->main_vector.data, spec->main_vector.size,
		               spec->main_vector.size_max, sizeof(t_part));
	}

	for (int i = 0; i < NUM_ADJ_PART; i++)
	{
		spec_merge_vectors(&spec->main_vector, &spec->incoming_part[i]);

		if(spec->gaspi_segm_offset_recv[i] >= 0)
		{
			int notif_id = NOTIFICATION_ID(OPPOSITE_DIR(i), 0, NOTIF_ID_PART_ACK(spec_id));
			if (i == PART_RIGHT || i == PART_LEFT)
				notif_id = NOTIFICATION_ID(OPPOSITE_DIR(i), region_id, NOTIF_ID_PART_ACK(spec_id));

			CHECK_GASPI_ERROR(gaspi_notify(PART_SEGMENT_ID(spec_id), adj_ranks[i],
			                               notif_id, COMM_PART_ACK, queue, GASPI_BLOCK));
		}
	}

	// Remove invalid particles
	for (int i = 0; i < spec->main_vector.size; ++i)
		if (spec->main_vector.data[i].invalid)
			spec->main_vector.data[i--] = spec->main_vector.data[--spec->main_vector.size];
}

/*********************************************************************************************
 Current deposition
 *********************************************************************************************/
// Current deposition (Esirkepov method)
void dep_current_esk(int ix0, int iy0, int di, int dj, t_part_data x0, t_part_data y0,
                     t_part_data x1, t_part_data y1, t_part_data qvx, t_part_data qvy,
                     t_part_data qvz, t_current *current)
{

	int i, j;
	t_fld S0x[4], S0y[4], S1x[4], S1y[4], DSx[4], DSy[4];
	t_fld Wx[16], Wy[16], Wz[16];

	S0x[0] = 0.0f;
	S0x[1] = 1.0f - x0;
	S0x[2] = x0;
	S0x[3] = 0.0f;

	S0y[0] = 0.0f;
	S0y[1] = 1.0f - y0;
	S0y[2] = y0;
	S0y[3] = 0.0f;

	for (i = 0; i < 4; i++)
	{
		S1x[i] = 0.0f;
		S1y[i] = 0.0f;
	}

	S1x[1 + di] = 1.0f - x1;
	S1x[2 + di] = x1;

	S1y[1 + dj] = 1.0f - y1;
	S1y[2 + dj] = y1;

	for (i = 0; i < 4; i++)
	{
		DSx[i] = S1x[i] - S0x[i];
		DSy[i] = S1y[i] - S0y[i];
	}

	for (j = 0; j < 4; j++)
	{
		for (i = 0; i < 4; i++)
		{
			Wx[i + 4 * j] = DSx[i] * (S0y[j] + DSy[j] / 2.0f);
			Wy[i + 4 * j] = DSy[j] * (S0x[i] + DSx[i] / 2.0f);
			Wz[i + 4 * j] = S0x[i] * S0y[j] + DSx[i] * S0y[j] / 2.0f + S0x[i] * DSy[j] / 2.0f
			                + DSx[i] * DSy[j] / 3.0f;
		}
	}

	// jx
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;

	for (j = 0; j < 4; j++)
	{
		t_fld c;

		c = -qvx * Wx[4 * j];
		J[ix0 - 1 + (iy0 - 1 + j) * nrow].x += c;
		for (i = 1; i < 4; i++)
		{
			c -= qvx * Wx[i + 4 * j];
			J[ix0 + i - 1 + (iy0 - 1 + j) * nrow].x += c;
		}
	}

	// jy
	for (i = 0; i < 4; i++)
	{
		t_fld c;

		c = -qvy * Wy[i];
		J[ix0 + i - 1 + (iy0 - 1) * nrow].y += c;
		for (j = 1; j < 4; j++)
		{
			c -= qvy * Wy[i + 4 * j];
			J[ix0 + i - 1 + (iy0 - 1 + j) * nrow].y += c;
		}
	}

	// jz
	for (j = 0; j < 4; j++)
	{
		for (i = 0; i < 4; i++)
		{
			J[ix0 + i - 1 + (iy0 - 1 + j) * nrow].z += qvz * Wz[i + 4 * j];
		}
	}

}

// Current deposition (adapted Villasenor-Bunemann method)
void dep_current_zamb(int ix, int iy, int di, int dj, float x0, float y0, float dx, float dy,
                      float qnx, float qny, float qvz, t_current *current)
{
	// Split the particle trajectory
	typedef struct {
		float x0, x1, y0, y1, dx, dy, qvz;
		int ix, iy;
	} t_vp;

	t_vp vp[3];
	int vnp = 1;

	// split
	vp[0].x0 = x0;
	vp[0].y0 = y0;

	vp[0].dx = dx;
	vp[0].dy = dy;

	vp[0].x1 = x0 + dx;
	vp[0].y1 = y0 + dy;

	vp[0].qvz = qvz / 2.0;

	vp[0].ix = ix;
	vp[0].iy = iy;

	// x split
	if (di != 0)
	{
		//int ib = ( di+1 )>>1;
		int ib = (di == 1);

		float delta = (x0 + dx - ib) / dx;

		// Add new particle
		vp[1].x0 = 1 - ib;
		vp[1].x1 = (x0 + dx) - di;
		vp[1].dx = dx * delta;
		vp[1].ix = ix + di;

		float ycross = y0 + dy * (1.0f - delta);

		vp[1].y0 = ycross;
		vp[1].y1 = vp[0].y1;
		vp[1].dy = dy * delta;
		vp[1].iy = iy;

		vp[1].qvz = vp[0].qvz * delta;

		// Correct previous particle
		vp[0].x1 = ib;
		vp[0].dx *= (1.0f - delta);

		vp[0].dy *= (1.0f - delta);
		vp[0].y1 = ycross;

		vp[0].qvz *= (1.0f - delta);

		vnp++;
	}

	// ysplit
	if (dj != 0)
	{
		int isy = 1 - (vp[0].y1 < 0.0f || vp[0].y1 >= 1.0f);

		// int jb = ( dj+1 )>>1;
		int jb = (dj == 1);

		// The static analyser gets confused by this but it is correct
		float delta = (vp[isy].y1 - jb) / vp[isy].dy;

		// Add new particle
		vp[vnp].y0 = 1 - jb;
		vp[vnp].y1 = vp[isy].y1 - dj;
		vp[vnp].dy = vp[isy].dy * delta;
		vp[vnp].iy = vp[isy].iy + dj;

		float xcross = vp[isy].x0 + vp[isy].dx * (1.0f - delta);

		vp[vnp].x0 = xcross;
		vp[vnp].x1 = vp[isy].x1;
		vp[vnp].dx = vp[isy].dx * delta;
		vp[vnp].ix = vp[isy].ix;

		vp[vnp].qvz = vp[isy].qvz * delta;

		// Correct previous particle
		vp[isy].y1 = jb;
		vp[isy].dy *= (1.0f - delta);

		vp[isy].dx *= (1.0f - delta);
		vp[isy].x1 = xcross;

		vp[isy].qvz *= (1.0f - delta);

		// Correct extra vp if needed
		if (isy < vnp - 1)
		{
			vp[1].y0 -= dj;
			vp[1].y1 -= dj;
			vp[1].iy += dj;
		}
		vnp++;
	}

	// Deposit virtual particle currents
	int k;
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;

	for (k = 0; k < vnp; k++)
	{
		float S0x[2], S1x[2], S0y[2], S1y[2];
		float wl1, wl2;
		float wp1[2], wp2[2];

		S0x[0] = 1.0f - vp[k].x0;
		S0x[1] = vp[k].x0;

		S1x[0] = 1.0f - vp[k].x1;
		S1x[1] = vp[k].x1;

		S0y[0] = 1.0f - vp[k].y0;
		S0y[1] = vp[k].y0;

		S1y[0] = 1.0f - vp[k].y1;
		S1y[1] = vp[k].y1;

		wl1 = qnx * vp[k].dx;
		wl2 = qny * vp[k].dy;

		wp1[0] = 0.5f * (S0y[0] + S1y[0]);
		wp1[1] = 0.5f * (S0y[1] + S1y[1]);

		wp2[0] = 0.5f * (S0x[0] + S1x[0]);
		wp2[1] = 0.5f * (S0x[1] + S1x[1]);

		J[vp[k].ix + nrow * vp[k].iy].x += wl1 * wp1[0];
		J[vp[k].ix + nrow * (vp[k].iy + 1)].x += wl1 * wp1[1];

		J[vp[k].ix + nrow * vp[k].iy].y += wl2 * wp2[0];
		J[vp[k].ix + 1 + nrow * vp[k].iy].y += wl2 * wp2[1];

		J[vp[k].ix + nrow * vp[k].iy].z += vp[k].qvz
		        * (S0x[0] * S0y[0] + S1x[0] * S1y[0] + (S0x[0] * S1y[0] - S1x[0] * S0y[0]) / 2.0f);
		J[vp[k].ix + 1 + nrow * vp[k].iy].z += vp[k].qvz
		        * (S0x[1] * S0y[0] + S1x[1] * S1y[0] + (S0x[1] * S1y[0] - S1x[1] * S0y[0]) / 2.0f);
		J[vp[k].ix + nrow * (vp[k].iy + 1)].z += vp[k].qvz
		        * (S0x[0] * S0y[1] + S1x[0] * S1y[1] + (S0x[0] * S1y[1] - S1x[0] * S0y[1]) / 2.0f);
		J[vp[k].ix + 1 + nrow * (vp[k].iy + 1)].z += vp[k].qvz
		        * (S0x[1] * S0y[1] + S1x[1] * S1y[1] + (S0x[1] * S1y[1] - S1x[1] * S0y[1]) / 2.0f);
	}
}

/*********************************************************************************************
 Particle advance
 *********************************************************************************************/

// EM fields interpolation
void interpolate_fld(const t_vfld *restrict const E, const t_vfld *restrict const B, const int nrow,
                     const int ix, const int iy, const t_fld x, const t_fld y,
                     t_vfld *restrict const Ep, t_vfld *restrict const Bp)
{
	const int ih = ix + ((x < 0.5f) ? -1 : 0);
	const int jh = iy + ((y < 0.5f) ? -1 : 0);

	const t_fld w1h = x + ((x < 0.5f) ? 0.5f : -0.5f);
	const t_fld w2h = y + ((y < 0.5f) ? 0.5f : -0.5f);

	Ep->x = (E[ih + iy * nrow].x * (1.0f - w1h) + E[ih + 1 + iy * nrow].x * w1h) * (1.0f - y)
			+ (E[ih + (iy + 1) * nrow].x * (1.0f - w1h) + E[ih + 1 + (iy + 1) * nrow].x * w1h) * y;
	Ep->y = (E[ix + jh * nrow].y * (1.0f - x) + E[ix + 1 + jh * nrow].y * x) * (1.0f - w2h)
			+ (E[ix + (jh + 1) * nrow].y * (1.0f - x) + E[ix + 1 + (jh + 1) * nrow].y * x) * w2h;
	Ep->z = (E[ix + iy * nrow].z * (1.0f - x) + E[ix + 1 + iy * nrow].z * x) * (1.0f - y)
			+ (E[ix + (iy + 1) * nrow].z * (1.0f - x) + E[ix + 1 + (iy + 1) * nrow].z * x) * y;

	Bp->x = (B[ix + jh * nrow].x * (1.0f - x) + B[ix + 1 + jh * nrow].x * x) * (1.0f - w2h)
			+ (B[ix + (jh + 1) * nrow].x * (1.0f - x) + B[ix + 1 + (jh + 1) * nrow].x * x) * w2h;
	Bp->y = (B[ih + iy * nrow].y * (1.0f - w1h) + B[ih + 1 + iy * nrow].y * w1h) * (1.0f - y)
			+ (B[ih + (iy + 1) * nrow].y * (1.0f - w1h) + B[ih + 1 + (iy + 1) * nrow].y * w1h) * y;
	Bp->z = (B[ih + jh * nrow].z * (1.0f - w1h) + B[ih + 1 + jh * nrow].z * w1h) * (1.0f - w2h)
			+ (B[ih + (jh + 1) * nrow].z * (1.0f - w1h) + B[ih + 1 + (jh + 1) * nrow].z * w1h) * w2h;
}

// Particle advance
void spec_advance(t_species *spec, const t_emf *emf, t_current *current,
                  const int region_limits[2][2], const int sim_nx[2])
{
	const t_part_data tem = 0.5 * spec->dt / spec->m_q;
	const t_part_data dt_dx = spec->dt / spec->dx[0];
	const t_part_data dt_dy = spec->dt / spec->dx[1];

	// Auxiliary values for current deposition
	const t_part_data qnx = spec->q * spec->dx[0] / spec->dt;
	const t_part_data qny = spec->q * spec->dx[1] / spec->dt;

	// Advance internal iteration number
	spec->iter++;
	const bool shift = (spec->iter * spec->dt) > (spec->dx[0] * (spec->n_move + 1));

	// Advance particles
	for (int i = 0; i < spec->main_vector.size; i++)
	{
		if (spec->main_vector.data[i].invalid) continue;

		t_vfld Ep, Bp;
		t_part_data utx, uty, utz;
		t_part_data ux, uy, uz, rg;
		t_part_data gtem, otsq;
		t_part_data qvz;
		t_part_data x0, y0, x1, y1;

		int local_ix, local_iy;
		int di, dj;
		float dx, dy;

		// Load particle info
		x0 = spec->main_vector.data[i].x;
		y0 = spec->main_vector.data[i].y;

		local_ix = spec->main_vector.data[i].ix - region_limits[0][0];
		local_iy = spec->main_vector.data[i].iy - region_limits[1][0];

		ux = spec->main_vector.data[i].ux;
		uy = spec->main_vector.data[i].uy;
		uz = spec->main_vector.data[i].uz;

		// Interpolate fields
		interpolate_fld(emf->E, emf->B, emf->nrow, local_ix, local_iy, x0, y0, &Ep, &Bp);

		// Advance u using Boris scheme
		Ep.x *= tem;
		Ep.y *= tem;
		Ep.z *= tem;

		utx = ux + Ep.x;
		uty = uy + Ep.y;
		utz = uz + Ep.z;

		// Perform first half of the rotation
		gtem = tem / sqrtf(1.0f + utx * utx + uty * uty + utz * utz);

		Bp.x *= gtem;
		Bp.y *= gtem;
		Bp.z *= gtem;

		otsq = 2.0f / (1.0f + Bp.x * Bp.x + Bp.y * Bp.y + Bp.z * Bp.z);

		ux = utx + uty * Bp.z - utz * Bp.y;
		uy = uty + utz * Bp.x - utx * Bp.z;
		uz = utz + utx * Bp.y - uty * Bp.x;

		// Perform second half of the rotation
		Bp.x *= otsq;
		Bp.y *= otsq;
		Bp.z *= otsq;

		utx += uy * Bp.z - uz * Bp.y;
		uty += uz * Bp.x - ux * Bp.z;
		utz += ux * Bp.y - uy * Bp.x;

		// Perform second half of electric field acceleration
		ux = utx + Ep.x;
		uy = uty + Ep.y;
		uz = utz + Ep.z;

		// Store new momenta
		spec->main_vector.data[i].ux = ux;
		spec->main_vector.data[i].uy = uy;
		spec->main_vector.data[i].uz = uz;

		// push particle
		rg = 1.0f / sqrtf(1.0f + ux * ux + uy * uy + uz * uz);

		dx = dt_dx * rg * ux;
		dy = dt_dy * rg * uy;

		x1 = x0 + dx;
		y1 = y0 + dy;

		di = LTRIM(x1);
		dj = LTRIM(y1);

		x1 -= di;
		y1 -= dj;

		qvz = spec->q * uz * rg;

		dep_current_zamb(local_ix, local_iy, di, dj, x0, y0, dx, dy, qnx, qny, qvz, current);

		// Store results
		spec->main_vector.data[i].x = x1;
		spec->main_vector.data[i].y = y1;
		spec->main_vector.data[i].ix += di;
		spec->main_vector.data[i].iy += dj;

		// First shift particle left (if applicable), then check for particles leaving the simulation space
		if (spec->moving_window)
		{
			if (shift) spec->main_vector.data[i].ix--;

			if ((spec->main_vector.data[i].ix < 0) || (spec->main_vector.data[i].ix >= sim_nx[0]))
			{
				spec->main_vector.data[i].invalid = true;
				continue;
			}
		}

		int target = -1;
		int iy = spec->main_vector.data[i].iy;
		int ix = spec->main_vector.data[i].ix;

		if (iy < region_limits[1][0])
		{
			if (ix < region_limits[0][0]) target = PART_DOWN_LEFT;
			else if (ix >= region_limits[0][1]) target = PART_DOWN_RIGHT;
			else target = PART_DOWN;
		} else if (iy >= region_limits[1][1])
		{
			if (ix < region_limits[0][0]) target = PART_UP_LEFT;
			else if (ix >= region_limits[0][1]) target = PART_UP_RIGHT;
			else target = PART_UP;
		} else
		{
			if (ix < region_limits[0][0]) target = PART_LEFT;
			else if (ix >= region_limits[0][1]) target = PART_RIGHT;
		}

		if (target >= 0)
		{
			if (!spec->moving_window)
				spec->main_vector.data[i].ix = PERIODIC_BOUNDARIES(ix, sim_nx[0]);
			spec->main_vector.data[i].iy = PERIODIC_BOUNDARIES(iy, sim_nx[1]);

			t_part_vector *out = spec->outgoing_part[target];
			out->data[out->size++] = spec->main_vector.data[i];
			spec->main_vector.data[i].invalid = true;
		}
	}

	if (spec->moving_window && shift)
	{
		// Increase moving window counter
		spec->n_move++;

		// Inject particles in the right edge of the simulation box
		const int range[][2] = { {sim_nx[0] - 1, sim_nx[0]},
		                         {region_limits[1][0], region_limits[1][1]}};
		spec_inject_particles(&spec->main_vector, range, region_limits, spec->ppc, &spec->density,
		                      spec->dx, spec->n_move, spec->ufl, spec->uth);
	}
}

/*********************************************************************************************
 Charge Deposition
 *********************************************************************************************/
// Deposit the particle charge over the simulation grid
void spec_deposit_charge(const t_species *spec, t_part_data *charge, const int nrow)
{
	// Charge array is expected to have 1 guard cell at the upper boundary
	t_part_data q = spec->q;

	for (int i = 0; i < spec->main_vector.size; i++)
	{
		if (spec->main_vector.data[i].invalid) continue;

		int ix = spec->main_vector.data[i].ix;
		int iy = spec->main_vector.data[i].iy;
		int idx = ix + iy * nrow;

		t_fld w1, w2;

		w1 = spec->main_vector.data[i].x;
		w2 = spec->main_vector.data[i].y;

		charge[idx] += (1.0f - w1) * (1.0f - w2) * q;
		charge[idx + 1] += (w1) * (1.0f - w2) * q;
		charge[idx + nrow] += (1.0f - w1) * (w2) * q;
		charge[idx + 1 + nrow] += (w1) * (w2) * q;
	}
}

/*********************************************************************************************
 Diagnostics
 *********************************************************************************************/

// Save the deposit particle charge in ZDF file
void spec_rep_charge(t_part_data *restrict charge, const int true_nx[2], const t_fld box[2],
                     const int iter_num, const float dt, const bool moving_window,
                     const char path[128])
{
	size_t buf_size = true_nx[0] * true_nx[1] * sizeof(t_part_data);
	t_part_data *restrict buf = malloc(buf_size);

	// Correct boundary values
	// x
	if (!moving_window)
		for (int j = 0; j < true_nx[1] + 1; j++)
			charge[0 + j * (true_nx[0] + 1)] += charge[true_nx[0] + j * (true_nx[0] + 1)];

	// y - Periodic boundaries
	for (int i = 0; i < true_nx[0] + 1; i++)
		charge[i] += charge[i + true_nx[1] * (true_nx[0] + 1)];

	t_part_data *restrict b = buf;
	t_part_data *restrict c = charge;

	for (int j = 0; j < true_nx[1]; j++)
	{
		for (int i = 0; i < true_nx[0]; i++)
			b[i] = c[i];

		b += true_nx[0];
		c += true_nx[0] + 1;
	}

	t_zdf_grid_axis axis[2];
	axis[0] = (t_zdf_grid_axis) {.min = 0.0, .max = box[0], .label = "x_1", .units = "c/\\omega_p"};
	axis[1] = (t_zdf_grid_axis) {.min = 0.0, .max = box[1], .label = "x_2", .units = "c/\\omega_p"};

	t_zdf_grid_info info = {.ndims = 2, .label = "charge", .units = "n_e", .axis = axis};
	info.nx[0] = true_nx[0];
	info.nx[1] = true_nx[1];

	t_zdf_iteration iter = {.n = iter_num, .t = iter_num * dt, .time_units = "1/\\omega_p"};
	zdf_save_grid(buf, &info, &iter, path);

	free(buf);
}

void spec_pha_axis(const t_species *spec, int i0, int np, int quant, float *axis)
{
	int i;

	switch (quant)
	{
		case X1:
			for (i = 0; i < np; i++)
				axis[i] = (spec->main_vector.data[i0 + i].x
						+ spec->main_vector.data[i0 + i].ix) * spec->dx[0];
			break;
		case X2:
			for (i = 0; i < np; i++)
				axis[i] = (spec->main_vector.data[i0 + i].y
						+ spec->main_vector.data[i0 + i].iy) * spec->dx[1];
			break;
		case U1:
			for (i = 0; i < np; i++)
				axis[i] = spec->main_vector.data[i0 + i].ux;
			break;
		case U2:
			for (i = 0; i < np; i++)
				axis[i] = spec->main_vector.data[i0 + i].uy;
			break;
		case U3:
			for (i = 0; i < np; i++)
				axis[i] = spec->main_vector.data[i0 + i].uz;
			break;
	}
}

const char* spec_pha_axis_units(int quant)
{
	switch (quant)
	{
		case X1:
		case X2:
			return ("c/\\omega_p");
			break;
		case U1:
		case U2:
		case U3:
			return ("m_e c");
	}
	return ("");
}

// Deposit the particles over the phase space
void spec_deposit_pha(const t_species *spec, const int rep_type, const int pha_nx[],
                      const float pha_range[][2], float *restrict buf)
{
	const int BUF_SIZE = 1024;
	float pha_x1[BUF_SIZE], pha_x2[BUF_SIZE];

	const int nrow = pha_nx[0];

	const int quant1 = rep_type & 0x000F;
	const int quant2 = (rep_type & 0x00F0) >> 4;

	const float x1min = pha_range[0][0];
	const float x2min = pha_range[1][0];

	const float rdx1 = pha_nx[0] / (pha_range[0][1] - pha_range[0][0]);
	const float rdx2 = pha_nx[1] / (pha_range[1][1] - pha_range[1][0]);

	for (int i = 0; i < spec->main_vector.size; i += BUF_SIZE)
	{
		int np = (i + BUF_SIZE > spec->main_vector.size) ? spec->main_vector.size - i : BUF_SIZE;

		spec_pha_axis(spec, i, np, quant1, pha_x1);
		spec_pha_axis(spec, i, np, quant2, pha_x2);

		for (int k = 0; k < np; k++)
		{
			float nx1 = (pha_x1[k] - x1min) * rdx1;
			float nx2 = (pha_x2[k] - x2min) * rdx2;

			int i1 = (int) (nx1 + 0.5f);
			int i2 = (int) (nx2 + 0.5f);

			float w1 = nx1 - i1 + 0.5f;
			float w2 = nx2 - i2 + 0.5f;

			int idx = i1 + nrow * i2;

			if (i2 >= 0 && i2 < pha_nx[1])
			{
				if (i1 >= 0 && i1 < pha_nx[0])
				{
					buf[idx] += (1.0f - w1) * (1.0f - w2) * spec->q;
				}

				if (i1 + 1 >= 0 && i1 + 1 < pha_nx[0])
				{
					buf[idx + 1] += w1 * (1.0f - w2) * spec->q;
				}
			}

			idx += nrow;
			if (i2 + 1 >= 0 && i2 + 1 < pha_nx[1])
			{

				if (i1 >= 0 && i1 < pha_nx[0])
				{
					buf[idx] += (1.0f - w1) * w2 * spec->q;
				}

				if (i1 + 1 >= 0 && i1 + 1 < pha_nx[0])
				{
					buf[idx + 1] += w1 * w2 * spec->q;
				}
			}

		}
	}
}

// Save the phase space in a ZDF file
void spec_rep_pha(const t_part_data *buffer, const int rep_type, const int pha_nx[],
                  const float pha_range[][2], const int iter_num, const float dt,
                  const char path[128])
{
	char const *const pha_ax_name[] = {"x1", "x2", "x3", "u1", "u2", "u3"};
	char pha_name[64];

	// save the data in hdf5 format
	int quant1 = rep_type & 0x000F;
	int quant2 = (rep_type & 0x00F0) >> 4;

	const char *pha_ax1_units = spec_pha_axis_units(quant1);
	const char *pha_ax2_units = spec_pha_axis_units(quant2);

	sprintf(pha_name, "%s%s", pha_ax_name[quant1 - 1], pha_ax_name[quant2 - 1]);

	t_zdf_grid_axis axis[2];
	axis[0] = (t_zdf_grid_axis) {.min = pha_range[0][0], .max = pha_range[0][1],
			                     .label = (char*) pha_ax_name[quant1 - 1],
			                     .units = (char*) pha_ax1_units};

	axis[1] = (t_zdf_grid_axis) {.min = pha_range[1][0], .max = pha_range[1][1],
			                     .label = (char*) pha_ax_name[quant2 - 1],
			                     .units = (char*) pha_ax2_units};

	t_zdf_grid_info info = {.ndims = 2, .label = pha_name, .units = "a.u.", .axis = axis};

	info.nx[0] = pha_nx[0];
	info.nx[1] = pha_nx[1];

	t_zdf_iteration iter = {.n = iter_num, .t = iter_num * dt, .time_units = "1/\\omega_p"};

	zdf_save_grid(buffer, &info, &iter, path);
}

// Calculate the energy of the particles
void spec_calculate_energy(t_species *spec)
{
	t_part_vector *restrict part = &spec->main_vector;
	spec->energy = 0;

	for (int i = 0; i < part->size; i++)
	{
		t_part_data usq = part->data[i].ux * part->data[i].ux + part->data[i].uy * part->data[i].uy
		                  + part->data[i].uz * part->data[i].uz;
		t_part_data gamma = sqrtf(1 + usq);
		spec->energy += usq / (gamma + 1.0);
	}
}
