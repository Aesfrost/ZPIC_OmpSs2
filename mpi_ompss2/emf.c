/*********************************************************************************************
 ZPIC
 emf.c

 Created by Ricardo Fonseca on 10/8/10.
 Modified by Nicolas Guidotti on 11/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include "emf.h"
#include "zdf.h"
#include "timer.h"
#include "task_management.h"

static double _emf_time = 0.0;
static MPI_Datatype MPI_VFLD = MPI_DATATYPE_NULL;

double emf_time(void)
{
	return _emf_time;
}

/*********************************************************************************************
 Constructor / Destructor
 *********************************************************************************************/
void emf_new(t_emf *emf, int nx[], t_fld box[], const float dt, const bool on_right_edge, const bool on_left_edge)
{
	int i;

	// Number of guard cells for linear interpolation
	int gc[2][2] = { {1, 2}, {1, 2}};

	// Allocate global arrays
	size_t size;

	size = (gc[0][0] + nx[0] + gc[0][1]) * (gc[1][0] + nx[1] + gc[1][1]) * sizeof(t_vfld);
	emf->total_size = (gc[0][0] + nx[0] + gc[0][1]) * (gc[1][0] + nx[1] + gc[1][1]);
	emf->overlap_size = (gc[0][0] + nx[0] + gc[0][1]) * (gc[1][0] + gc[1][1]);

	emf->E_buf = malloc(size);
	emf->B_buf = malloc(size);

	assert(emf->E_buf && emf->B_buf);

	// zero fields
	memset(emf->E_buf, 0, size);
	memset(emf->B_buf, 0, size);

	// store nx and gc values
	for (i = 0; i < 2; i++)
	{
		emf->nx[i] = nx[i];
		emf->gc[i][0] = gc[i][0];
		emf->gc[i][1] = gc[i][1];
	}
	emf->nrow = gc[0][0] + nx[0] + gc[0][1];

	// store time step values
	emf->dt = dt;

	// Make E and B point to cell [0][0]
	emf->E = emf->E_buf + gc[0][0] + gc[1][0] * emf->nrow;
	emf->B = emf->B_buf + gc[0][0] + gc[1][0] * emf->nrow;

	// Set cell sizes and box limits
	for (i = 0; i < 2; i++)
	{
		emf->box[i] = box[i];
		emf->dx[i] = box[i] / nx[i];
	}

	// Set time step
	emf->dt = dt;

	// Reset iteration number
	emf->iter = 0;

	// Reset moving window information
	emf->moving_window = false;
	emf->n_move = 0;

	emf->on_left_edge = on_left_edge;
	emf->on_right_edge = on_right_edge;

	if(MPI_VFLD == MPI_DATATYPE_NULL)
	{
		CHECK_MPI_ERROR(MPI_Type_contiguous(3, MPI_FLOAT, &MPI_VFLD));
		CHECK_MPI_ERROR(MPI_Type_commit(&MPI_VFLD));
	}

	// Reset all MPI requests
	emf->num_mpi_requests = 0;
	for (int i = 0; i < 8; ++i)
		emf->mpi_requests[i] = MPI_REQUEST_NULL;
}

void emf_delete(t_emf *emf)
{
	free(emf->E_buf);
	free(emf->B_buf);

	emf->E_buf = NULL;
	emf->B_buf = NULL;

	for (int i = 0; i < NUM_ADJ_GRID; ++i)
	{
		if(emf->inter_proc_comm[i])
		{
			free(emf->send_B[i]);
			free(emf->receive_B[i]);
			free(emf->send_E[i]);
			free(emf->receive_E[i]);
		}
	}

	if(MPI_VFLD != MPI_DATATYPE_NULL)
	{
		CHECK_MPI_ERROR(MPI_Type_free(&MPI_VFLD));
		MPI_VFLD = MPI_DATATYPE_NULL;
	}
}

/*********************************************************************************************
 Laser Pulses
 *********************************************************************************************/

t_fld gauss_phase(const t_emf_laser *const laser, const t_fld z, const t_fld r)
{
	t_fld z0 = laser->omega0 * (laser->W0 * laser->W0) / 2;
	t_fld rho2 = r * r;
	t_fld curv = rho2 * z / (z0 * z0 + z * z);
	t_fld rWl2 = (z0 * z0) / (z0 * z0 + z * z);
	t_fld gouy_shift = atan2(z, z0);

	return sqrt(sqrt(rWl2)) * exp(-rho2 * rWl2 / (laser->W0 * laser->W0))
	       * cos(laser->omega0 * (z + curv) - gouy_shift);
}

t_fld lon_env(const t_emf_laser *const laser, const t_fld z)
{
	if (z > laser->start)
	{
		// Ahead of laser
		return 0.0;
	} else if (z > laser->start - laser->rise)
	{
		// Laser rise
		t_fld csi = z - laser->start;
		t_fld e = sin(M_PI_2 * csi / laser->rise);
		return e * e;
	} else if (z > laser->start - (laser->rise + laser->flat))
	{
		// Flat-top
		return 1.0;
	} else if (z > laser->start - (laser->rise + laser->flat + laser->fall))
	{
		// Laser fall
		t_fld csi = z - (laser->start - laser->rise - laser->flat - laser->fall);
		t_fld e = sin(M_PI_2 * csi / laser->fall);
		return e * e;
	}

	// Before laser
	return 0.0;
}

void div_corr_x(t_vfld *restrict E, t_vfld *restrict B, const int nx[2], const int nrow,
                const float dx[2])
{
	const double dx_dy = dx[0] / dx[1];

	for (int j = 0; j < nx[1]; j++)
	{
		double ex = 0.0;
		double bx = 0.0;
		for (int i = nx[0] - 1; i >= 0; i--)
		{
			ex += dx_dy * (E[i + 1 + j * nrow].y - E[i + 1 + (j - 1) * nrow].y);
			E[i + j * nrow].x = ex;

			bx += dx_dy * (B[i + (j + 1) * nrow].y - B[i + j * nrow].y);
			B[i + j * nrow].x = bx;
		}

	}
}

void emf_add_laser(t_emf_laser *laser, t_vfld *restrict E, t_vfld *restrict B, const int nx[2],
                   const int nrow, const float dx[2], const int gc[2][2])
{
	// Validate laser parameters
	if (laser->fwhm != 0)
	{
		if (laser->fwhm <= 0)
		{
			fprintf(stderr, "Invalid laser FWHM, must be > 0, aborting.\n");
			exit(-1);
		}
		// The fwhm parameter overrides the rise/flat/fall parameters
		laser->rise = laser->fwhm;
		laser->fall = laser->fwhm;
		laser->flat = 0.;
	}

	if (laser->rise <= 0)
	{
		fprintf(stderr, "Invalid laser RISE, must be > 0, aborting.\n");
		exit(-1);
	}

	if (laser->flat < 0)
	{
		fprintf(stderr, "Invalid laser FLAT, must be >= 0, aborting.\n");
		exit(-1);
	}

	if (laser->fall <= 0)
	{
		fprintf(stderr, "Invalid laser FALL, must be > 0, aborting.\n");
		exit(-1);
	}

	// Launch laser
	int i, j;

	t_fld r_center, z, z_2, r, r_2;
	t_fld amp, lenv, lenv_2, k;
	t_fld cos_pol, sin_pol;

	r_center = laser->axis;
	amp = laser->omega0 * laser->a0;

	cos_pol = cos(laser->polarization);
	sin_pol = sin(laser->polarization);

	switch (laser->type)
	{
		case PLANE:
			k = laser->omega0;

			for (i = 0; i < nx[0]; i++)
			{
				z = i * dx[0];
				z_2 = z + dx[0] / 2;

				lenv = amp * lon_env(laser, z);
				lenv_2 = amp * lon_env(laser, z_2);

				for (j = 0; j < nx[1]; j++)
				{
					// E[i + j*nrow].x += 0.0
					E[i + j * nrow].y += +lenv * cos(k * z) * cos_pol;
					E[i + j * nrow].z += +lenv * cos(k * z) * sin_pol;

					// E[i + j*nrow].x += 0.0
					B[i + j * nrow].y += -lenv_2 * cos(k * z_2) * sin_pol;
					B[i + j * nrow].z += +lenv_2 * cos(k * z_2) * cos_pol;
				}
			}
			break;

		case GAUSSIAN:

			for (i = 0; i < nx[0]; i++)
			{
				z = i * dx[0];
				z_2 = z + dx[0] / 2;

				lenv = amp * lon_env(laser, z);
				lenv_2 = amp * lon_env(laser, z_2);

				for (j = 0; j < nx[1]; j++)
				{
					r = j * dx[1] - r_center;
					r_2 = r + dx[1] / 2;

					// E[i + j*nrow].x += 0.0
					E[i + j * nrow].y += +lenv * gauss_phase(laser, z, r_2) * cos_pol;
					E[i + j * nrow].z += +lenv * gauss_phase(laser, z, r) * sin_pol;

					// B[i + j*nrow].x += 0.0
					B[i + j * nrow].y += -lenv_2 * gauss_phase(laser, z_2, r) * sin_pol;
					B[i + j * nrow].z += +lenv_2 * gauss_phase(laser, z_2, r_2) * cos_pol;

				}
			}
			div_corr_x(E, B, nx, nrow, dx);

			break;
		default:
			break;
	}

	// Set guard cell values
	emf_update_gc_serial(E, B, nx, nrow, gc);
}

/*********************************************************************************************
 Diagnostics
 *********************************************************************************************/

// Reconstruct the simulation grid from all regions (eletric/magnetic field for a given direction)
void emf_reconstruct_global_buffer(const t_emf *emf, float *global_buffer, const int offset_y,
                                   const int offset_x, const int sim_nrow, const char field,
                                   const char fc)
{
	t_vfld *restrict f = NULL;
	float *restrict p = global_buffer + offset_x + offset_y * sim_nrow;

	switch (field)
	{
		case EFLD:
			f = emf->E;
			break;
		case BFLD:
			f = emf->B;
			break;
		default:
			fprintf(stderr, "Invalid field type selected, returning\n");
			break;
	}

	switch (fc)
	{
		case 0:
			for (int j = 0; j < emf->nx[1]; j++)
			{
				for (int i = 0; i < emf->nx[0]; i++)
					p[i] = f[i].x;
				p += sim_nrow;
				f += emf->nrow;
			}
			break;
		case 1:
			for (int j = 0; j < emf->nx[1]; j++)
			{
				for (int i = 0; i < emf->nx[0]; i++)
					p[i] = f[i].y;
				p += sim_nrow;
				f += emf->nrow;
			}
			break;
		case 2:
			for (int j = 0; j < emf->nx[1]; j++)
			{
				for (int i = 0; i < emf->nx[0]; i++)
					p[i] = f[i].z;
				p += sim_nrow;
				f += emf->nrow;
			}
			break;
		default:
			fprintf(stderr, "Invalid field component selected, returning\n");
			return;
	}
}

// Save the reconstructed buffer in a ZDF file
void emf_report(const float *restrict global_buffer, const float box[2], const int true_nx[2],
                const int iter, const float dt, const char field, const char fc,
                const char path[128])
{
	char vfname[3];

	// Choose field to save
	switch (field)
	{
		case EFLD:
			vfname[0] = 'E';
			break;
		case BFLD:
			vfname[0] = 'B';
			break;
		default:
			fprintf(stderr, "Invalid field type selected, returning\n");
			return;
	}

	switch (fc)
	{
		case 0:
			vfname[1] = '1';
			break;
		case 1:
			vfname[1] = '2';
			break;
		case 2:
			vfname[1] = '3';
			break;
		default:
			fprintf(stderr, "Invalid field component selected, returning\n");
			return;
	}
	vfname[2] = 0;

	t_zdf_grid_axis axis[2];
	axis[0] = (t_zdf_grid_axis) {.min = 0.0, .max = box[0], .label = "x_1", .units = "c/\\omega_p"};

	axis[1] = (t_zdf_grid_axis) {.min = 0.0, .max = box[1], .label = "x_2", .units = "c/\\omega_p"};

	t_zdf_grid_info info = {.ndims = 2, .label = vfname, .units = "m_e c \\omega_p e^{-1}",
	                        .axis = axis};

	info.nx[0] = true_nx[0];
	info.nx[1] = true_nx[1];

	t_zdf_iteration iteration = {.n = iter, .t = iter * dt, .time_units = "1/\\omega_p"};

	zdf_save_grid(global_buffer, &info, &iteration, path);

}

// Calculate the EMF energy
double emf_get_energy(t_emf *emf)
{
	t_vfld *const restrict E = emf->E;
	t_vfld *const restrict B = emf->B;
	double result = 0;

	for (unsigned int i = 0; i < emf->nx[0] * emf->nx[1]; i++)
	{
		result += 2 * E[i].x * E[i].x;
		result += E[i].y * E[i].y;
		result += E[i].z * E[i].z;
		result += B[i].x * B[i].x;
		result += B[i].y * B[i].y;
		result += B[i].z * B[i].z;
	}

	return result * 0.5 * emf->dx[0] * emf->dx[1];
}

/*********************************************************************************************
 Comunication
 *********************************************************************************************/
// Set the overlap zone between regions (upper zone only)
void emf_link_adj_regions(t_emf *emf, t_emf *emf_down, t_emf *emf_up)
{
	const int segm_nrow = emf->gc[0][0] + emf->gc[0][1];

	for (int dir = 0; dir < 4; dir++)
	{
		switch (dir)
		{
			case GRID_DOWN:
				if (emf_down)   // The adjacent region (bottom) is in the same process
				{
					emf->send_E[dir] = emf_down->E_buf + emf_down->nx[1] * emf_down->nrow;
					emf->receive_E[dir] = emf->send_E[dir];
					emf->send_B[dir] = emf_down->B_buf + emf_down->nx[1] * emf_down->nrow;
					emf->receive_B[dir] = emf->send_B[dir];
					emf->inter_proc_comm[dir] = false;
				}else
				{
					emf->send_E[dir] = calloc(emf->overlap_size, sizeof(t_vfld));
					emf->receive_E[dir] = calloc(emf->overlap_size, sizeof(t_vfld));
					emf->send_B[dir] = calloc(emf->overlap_size, sizeof(t_vfld));
					emf->receive_B[dir] = calloc(emf->overlap_size, sizeof(t_vfld));
					emf->inter_proc_comm[dir] = true;
				}
				break;

			case GRID_UP:
				if (emf_up)   // The adjacent region (upper) is in the same process
				{
					emf->send_E[dir] = emf_up->E_buf;
					emf->receive_E[dir] = emf->send_E[dir];
					emf->send_B[dir] = emf_up->B_buf;
					emf->receive_B[dir] = emf->send_B[dir];
					emf->inter_proc_comm[dir] = false;
				}else
				{
					emf->send_E[dir] = calloc(emf->overlap_size, sizeof(t_vfld));
					emf->receive_E[dir] = calloc(emf->overlap_size, sizeof(t_vfld));
					emf->send_B[dir] = calloc(emf->overlap_size, sizeof(t_vfld));
					emf->receive_B[dir] = calloc(emf->overlap_size, sizeof(t_vfld));
					emf->inter_proc_comm[dir] = true;
				}
				break;

			default:   // GRID_LEFT or GRID_RIGHT
				emf->send_E[dir] = calloc(segm_nrow * emf->nx[1], sizeof(t_vfld));
				emf->receive_E[dir] = calloc(segm_nrow * emf->nx[1], sizeof(t_vfld));
				emf->send_B[dir] = calloc(segm_nrow * emf->nx[1], sizeof(t_vfld));
				emf->receive_B[dir] = calloc(segm_nrow * emf->nx[1], sizeof(t_vfld));
				emf->inter_proc_comm[dir] = true;
				break;
		}
	}
}

void emf_exchange_gc_x(t_emf *emf, const int region_id, const unsigned int adj_ranks[NUM_ADJ_GRID])
{
	t_vfld *restrict E = emf->E;
	t_vfld *restrict B = emf->B;
	t_vfld *restrict E_left = emf->send_E[GRID_LEFT];
	t_vfld *restrict B_left = emf->send_B[GRID_LEFT];
	t_vfld *restrict E_right = emf->send_E[GRID_RIGHT];
	t_vfld *restrict B_right = emf->send_B[GRID_RIGHT];

	const int nrow = emf->nrow;
	const int segm_nrow = emf->gc[0][0] + emf->gc[0][1];

	emf->num_mpi_requests = 0;
	for (int i = 0; i < 8; ++i)
		emf->mpi_requests[i] = MPI_REQUEST_NULL;

	if (!emf->moving_window || !emf->on_left_edge)
	{
		for (int j = 0; j < emf->nx[1]; ++j)
		{
			for (int i = 0; i < segm_nrow; ++i)
			{
				E_left[i + j * segm_nrow] = E[i - emf->gc[0][0] + j * nrow];
				B_left[i + j * segm_nrow] = B[i - emf->gc[0][0] + j * nrow];
			}
		}

		CHECK_MPI_ERROR(MPI_Isend(E_left,
		                          emf->nx[1] * segm_nrow,
		                          MPI_VFLD,
		                          adj_ranks[GRID_LEFT],
		                          CREATE_MPI_TAG(GRID_RIGHT, region_id, MPI_TAG_E),
		                          MPI_COMM_WORLD,
		                          &emf->mpi_requests[emf->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Isend(B_left,
		                          emf->nx[1] * segm_nrow,
		                          MPI_VFLD,
		                          adj_ranks[GRID_LEFT],
		                          CREATE_MPI_TAG(GRID_RIGHT, region_id, MPI_TAG_B),
		                          MPI_COMM_WORLD,
		                          &emf->mpi_requests[emf->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Irecv(emf->receive_E[GRID_LEFT],
		                         emf->nx[1] * segm_nrow,
		                         MPI_VFLD,
		                         adj_ranks[GRID_LEFT],
		                         CREATE_MPI_TAG(GRID_LEFT, region_id, MPI_TAG_E),
		                         MPI_COMM_WORLD,
		                         &emf->mpi_requests[emf->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Irecv(emf->receive_B[GRID_LEFT],
		                         emf->nx[1] * segm_nrow,
		                         MPI_VFLD,
		                         adj_ranks[GRID_LEFT],
		                         CREATE_MPI_TAG(GRID_LEFT, region_id, MPI_TAG_B),
		                         MPI_COMM_WORLD,
		                         &emf->mpi_requests[emf->num_mpi_requests++]));
	}

	if (!emf->moving_window || !emf->on_right_edge)
	{
		for (int j = 0; j < emf->nx[1]; ++j)
		{
			for (int i = 0; i < segm_nrow; ++i)
			{
				E_right[i + j * segm_nrow] = E[emf->nx[0] - 1 + i + j * nrow];
				B_right[i + j * segm_nrow] = B[emf->nx[0] - 1 + i + j * nrow];
			}
		}

		CHECK_MPI_ERROR(MPI_Isend(E_right,
		                          emf->nx[1] * segm_nrow,
		                          MPI_VFLD,
		                          adj_ranks[GRID_RIGHT],
		                          CREATE_MPI_TAG(GRID_LEFT, region_id, MPI_TAG_E),
		                          MPI_COMM_WORLD,
		                          &emf->mpi_requests[emf->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Isend(B_right,
		                          emf->nx[1] * segm_nrow,
		                          MPI_VFLD,
		                          adj_ranks[GRID_RIGHT],
		                          CREATE_MPI_TAG(GRID_LEFT, region_id, MPI_TAG_B),
		                          MPI_COMM_WORLD,
		                          &emf->mpi_requests[emf->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Irecv(emf->receive_E[GRID_RIGHT],
		                         emf->nx[1] * segm_nrow,
		                         MPI_VFLD,
		                         adj_ranks[GRID_RIGHT],
		                         CREATE_MPI_TAG(GRID_RIGHT, region_id, MPI_TAG_E),
		                         MPI_COMM_WORLD,
		                         &emf->mpi_requests[emf->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Irecv(emf->receive_B[GRID_RIGHT],
		                         emf->nx[1] * segm_nrow,
		                         MPI_VFLD,
		                         adj_ranks[GRID_RIGHT],
		                         CREATE_MPI_TAG(GRID_RIGHT, region_id, MPI_TAG_B),
		                         MPI_COMM_WORLD,
		                         &emf->mpi_requests[emf->num_mpi_requests++]));
	}
}

void emf_update_gc_x(t_emf *emf)
{
	const int nrow = emf->nrow;
	const int segm_nrow = emf->gc[0][0] + emf->gc[0][1];

	t_vfld *restrict E = emf->E_buf + emf->gc[1][0] * nrow;
	t_vfld *restrict B = emf->B_buf + emf->gc[1][0] * nrow;
	t_vfld *restrict E_left = emf->receive_E[GRID_LEFT];
	t_vfld *restrict B_left = emf->receive_B[GRID_LEFT];
	t_vfld *restrict E_right = emf->receive_E[GRID_RIGHT];
	t_vfld *restrict B_right = emf->receive_B[GRID_RIGHT];

	mpi_wait_async_comm(emf->mpi_requests, emf->num_mpi_requests);

	if (emf->moving_window && emf->shift_window_iter)
	{
		if (!emf->on_right_edge)
		{
			for (int j = 0; j < emf->nx[1]; ++j)
			{
				for (int i = 0; i < segm_nrow; ++i)
				{
					E[emf->nx[0] + i + j * nrow] = E_right[i + j * segm_nrow];
					B[emf->nx[0] + i + j * nrow] = B_right[i + j * segm_nrow];
				}
			}
		}
	} else
	{
		for (int j = 0; j < emf->nx[1]; ++j)
		{
			if (!emf->moving_window || !emf->on_left_edge)
			{
				for (int i = 0; i < emf->gc[0][0]; ++i)
				{
					E[i + j * nrow] = E_left[i + j * segm_nrow];
					B[i + j * nrow] = B_left[i + j * segm_nrow];
				}
			}

			if (!emf->moving_window || !emf->on_right_edge)
			{
				for (int i = emf->gc[0][0]; i < segm_nrow; ++i)
				{
					E[emf->nx[0] + i + j * nrow] = E_right[i + j * segm_nrow];
					B[emf->nx[0] + i + j * nrow] = B_right[i + j * segm_nrow];
				}
			}
		}
	}
}

void emf_exchange_gc_y(t_emf *emf, const unsigned int adj_ranks[NUM_ADJ_GRID])
{
	const int nrow = emf->nrow;

	emf->num_mpi_requests = 0;
	for (int i = 0; i < 8; ++i)
		emf->mpi_requests[i] = MPI_REQUEST_NULL;

	if (emf->inter_proc_comm[GRID_DOWN])
	{
		memcpy(emf->send_E[GRID_DOWN], emf->E_buf, emf->overlap_size * sizeof(t_vfld));
		memcpy(emf->send_B[GRID_DOWN], emf->B_buf, emf->overlap_size * sizeof(t_vfld));

		CHECK_MPI_ERROR(MPI_Isend(emf->send_E[GRID_DOWN],
		                          emf->overlap_size,
		                          MPI_VFLD,
		                          adj_ranks[GRID_DOWN],
		                          CREATE_MPI_TAG(GRID_UP, 0, MPI_TAG_E),
		                          MPI_COMM_WORLD,
		                          &emf->mpi_requests[emf->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Isend(emf->send_B[GRID_DOWN],
		                          emf->overlap_size,
		                          MPI_VFLD,
		                          adj_ranks[GRID_DOWN],
		                          CREATE_MPI_TAG(GRID_UP, 0, MPI_TAG_B),
		                          MPI_COMM_WORLD,
		                          &emf->mpi_requests[emf->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Irecv(emf->receive_E[GRID_DOWN],
								 emf->overlap_size,
								 MPI_VFLD,
								 adj_ranks[GRID_DOWN],
								 CREATE_MPI_TAG(GRID_DOWN, 0, MPI_TAG_E),
								 MPI_COMM_WORLD,
								 &emf->mpi_requests[emf->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Irecv(emf->receive_B[GRID_DOWN],
								 emf->overlap_size,
								 MPI_VFLD,
								 adj_ranks[GRID_DOWN],
								 CREATE_MPI_TAG(GRID_DOWN, 0, MPI_TAG_B),
								 MPI_COMM_WORLD,
								 &emf->mpi_requests[emf->num_mpi_requests++]));
	}

	if (emf->inter_proc_comm[GRID_UP])
	{
		memcpy(emf->send_E[GRID_UP], emf->E_buf + emf->nx[1] * nrow,
		       emf->overlap_size * sizeof(t_vfld));
		memcpy(emf->send_B[GRID_UP], emf->B_buf + emf->nx[1] * nrow,
		       emf->overlap_size * sizeof(t_vfld));

		CHECK_MPI_ERROR(MPI_Isend(emf->send_E[GRID_UP],
		                          emf->overlap_size,
		                          MPI_VFLD,
		                          adj_ranks[GRID_UP],
		                          CREATE_MPI_TAG(GRID_DOWN, 0, MPI_TAG_E),
		                          MPI_COMM_WORLD,
		                          &emf->mpi_requests[emf->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Isend(emf->send_B[GRID_UP],
		                          emf->overlap_size,
		                          MPI_VFLD,
		                          adj_ranks[GRID_UP],
		                          CREATE_MPI_TAG(GRID_DOWN, 0, MPI_TAG_B),
		                          MPI_COMM_WORLD,
		                          &emf->mpi_requests[emf->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Irecv(emf->receive_E[GRID_UP],
								 emf->overlap_size,
								 MPI_VFLD,
								 adj_ranks[GRID_UP],
								 CREATE_MPI_TAG(GRID_UP, 0, MPI_TAG_E),
								 MPI_COMM_WORLD,
								 &emf->mpi_requests[emf->num_mpi_requests++]));

		CHECK_MPI_ERROR(MPI_Irecv(emf->receive_B[GRID_UP],
								 emf->overlap_size,
								 MPI_VFLD,
								 adj_ranks[GRID_UP],
								 CREATE_MPI_TAG(GRID_UP, 0, MPI_TAG_B),
								 MPI_COMM_WORLD,
								 &emf->mpi_requests[emf->num_mpi_requests++]));
	}
}

void emf_update_gc_y(t_emf *emf)
{
	const int nrow = emf->nrow;
	t_vfld *restrict E = emf->E_buf;
	t_vfld *restrict B = emf->B_buf;
	t_vfld *restrict E_up = emf->receive_E[GRID_UP];
	t_vfld *restrict B_up = emf->receive_B[GRID_UP];
	t_vfld *restrict E_down = emf->receive_E[GRID_DOWN];
	t_vfld *restrict B_down = emf->receive_B[GRID_DOWN];

	mpi_wait_async_comm(emf->mpi_requests, emf->num_mpi_requests);

	memcpy(E, E_down, emf->gc[1][0] * nrow * sizeof(t_vfld));
	memcpy(B, B_down, emf->gc[1][0] * nrow * sizeof(t_vfld));
	memcpy(E + (emf->gc[1][0] + emf->nx[1]) * nrow, E_up + emf->gc[1][0] * nrow,
	       emf->gc[1][1] * nrow * sizeof(t_vfld));
	memcpy(B + (emf->gc[1][0] + emf->nx[1]) * nrow, B_up + emf->gc[1][0] * nrow,
	       emf->gc[1][1] * nrow * sizeof(t_vfld));

}

void emf_update_gc_serial(t_vfld *restrict E, t_vfld *restrict B, const int nx[2], const int nrow,
                          const int gc[2][2])
{
	// x
	for (int j = -gc[1][0]; j < nx[1] + gc[1][1]; j++)
	{
		// lower
		for (int i = -gc[0][0]; i < 0; i++)
		{
			E[i + j * nrow].x = E[nx[0] + i + j * nrow].x;
			E[i + j * nrow].y = E[nx[0] + i + j * nrow].y;
			E[i + j * nrow].z = E[nx[0] + i + j * nrow].z;

			B[i + j * nrow].x = B[nx[0] + i + j * nrow].x;
			B[i + j * nrow].y = B[nx[0] + i + j * nrow].y;
			B[i + j * nrow].z = B[nx[0] + i + j * nrow].z;
		}

		// upper
		for (int i = 0; i < gc[0][1]; i++)
		{
			E[nx[0] + i + j * nrow].x = E[i + j * nrow].x;
			E[nx[0] + i + j * nrow].y = E[i + j * nrow].y;
			E[nx[0] + i + j * nrow].z = E[i + j * nrow].z;

			B[nx[0] + i + j * nrow].x = B[i + j * nrow].x;
			B[nx[0] + i + j * nrow].y = B[i + j * nrow].y;
			B[nx[0] + i + j * nrow].z = B[i + j * nrow].z;
		}

	}

	// y
	for (int i = -gc[0][0]; i < nx[0] + gc[0][1]; i++)
	{
		// lower
		for (int j = -gc[1][0]; j < 0; j++)
		{
			E[i + j * nrow].x = E[i + (nx[1] + j) * nrow].x;
			E[i + j * nrow].y = E[i + (nx[1] + j) * nrow].y;
			E[i + j * nrow].z = E[i + (nx[1] + j) * nrow].z;

			B[i + j * nrow].x = B[i + (nx[1] + j) * nrow].x;
			B[i + j * nrow].y = B[i + (nx[1] + j) * nrow].y;
			B[i + j * nrow].z = B[i + (nx[1] + j) * nrow].z;
		}

		// upper
		for (int j = 0; j < gc[1][1]; j++)
		{
			E[i + (nx[1] + j) * nrow].x = E[i + j * nrow].x;
			E[i + (nx[1] + j) * nrow].y = E[i + j * nrow].y;
			E[i + (nx[1] + j) * nrow].z = E[i + j * nrow].z;

			B[i + (nx[1] + j) * nrow].x = B[i + j * nrow].x;
			B[i + (nx[1] + j) * nrow].y = B[i + j * nrow].y;
			B[i + (nx[1] + j) * nrow].z = B[i + j * nrow].z;
		}
	}
}

// Move the simulation window
void emf_move_window(t_emf *emf)
{
	if ((emf->iter * emf->dt) > emf->dx[0] * (emf->n_move + 1))
	{
		const int nrow = emf->nrow;

		t_vfld *const restrict E = emf->E;
		t_vfld *const restrict B = emf->B;

		const t_vfld zero_fld = {0., 0., 0.};

		// Shift data left 1 cell and zero rightmost cells
		for (int j = 0; j < emf->nx[1]; j++)
		{
			for (int i = -emf->gc[0][0]; i < emf->nx[0]; i++)
			{
				E[i + j * nrow] = E[i + j * nrow + 1];
				B[i + j * nrow] = B[i + j * nrow + 1];
			}

			if (emf->on_right_edge)
			{
				for (int i = emf->nx[0] - 1; i < emf->nx[0] + emf->gc[0][1]; i++)
				{
					E[i + j * nrow] = zero_fld;
					B[i + j * nrow] = zero_fld;
				}
			}
		}

		// Increase moving window counter
		emf->n_move++;
		emf->shift_window_iter = true;
	}
}

/*********************************************************************************************
 Field solver
 *********************************************************************************************/

void yee_b(t_emf *emf, const float dt)
{
	t_vfld *const restrict B = emf->B;
	const t_vfld *const restrict E = emf->E;

	t_fld dt_dx = dt / emf->dx[0];
	t_fld dt_dy = dt / emf->dx[1];

	// Canonical implementation
	const int nrow = emf->nrow;
	for (int j = -1; j <= emf->nx[1]; j++)
	{
		for (int i = -1; i <= emf->nx[0]; i++)
		{
			B[i + j * nrow].x += -dt_dy * (E[i + (j + 1) * nrow].z - E[i + j * nrow].z);
			B[i + j * nrow].y += dt_dx * (E[(i + 1) + j * nrow].z - E[i + j * nrow].z);
			B[i + j * nrow].z += -dt_dx * (E[(i + 1) + j * nrow].y - E[i + j * nrow].y)
								+ dt_dy * (E[i + (j + 1) * nrow].x - E[i + j * nrow].x);
		}
	}
}

void yee_e(t_emf *emf, const t_current *current, const float dt)
{
	t_fld dt_dx = dt / emf->dx[0];
	t_fld dt_dy = dt / emf->dx[1];

	t_vfld *const restrict E = emf->E;
	const t_vfld *const restrict B = emf->B;
	const t_vfld *const restrict J = current->J;

	// Canonical implementation
	const int nrow_e = emf->nrow;
	const int nrow_j = current->nrow;

	for (int j = 0; j <= emf->nx[1] + 1; j++)
	{
		for (int i = 0; i <= emf->nx[0] + 1; i++)
		{
			E[i + j * nrow_e].x += (dt_dy * (B[i + j * nrow_e].z - B[i + (j - 1) * nrow_e].z))
			        				- dt * J[i + j * nrow_j].x;

			E[i + j * nrow_e].y += (-dt_dx * (B[i + j * nrow_e].z - B[(i - 1) + j * nrow_e].z))
			        				- dt * J[i + j * nrow_j].y;

			E[i + j * nrow_e].z += (dt_dx * (B[i + j * nrow_e].y - B[(i - 1) + j * nrow_e].y)
									- dt_dy * (B[i + j * nrow_e].x - B[i + (j - 1) * nrow_e].x))
									- dt * J[i + j * nrow_j].z;

		}
	}
}

// Perform the local integration of the fields (and post processing)
void emf_advance(t_emf *emf, const t_current *current)
{
	const float dt = emf->dt;

	// Advance EM field using Yee algorithm modified for having E and B time centered
	yee_b(emf, dt / 2.0f);
	yee_e(emf, current, dt);
	yee_b(emf, dt / 2.0f);

	// Advance internal iteration number
	emf->iter += 1;

	emf->shift_window_iter = false;
	if (emf->moving_window)
		emf_move_window(emf);
}

