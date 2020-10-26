/*********************************************************************************************
 ZPIC
 emf.c

 Created by Ricardo Fonseca on 10/8/10.
 Modified by Nicolas Guidotti on 11/06/20

 Copyright 2010 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include "emf.h"
#include "zdf.h"
#include "timer.h"
#include "csv_handler.h"

static double _emf_time = 0.0;

double emf_time(void)
{
	return _emf_time;
}

/*********************************************************************************************
 Constructor / Destructor
 *********************************************************************************************/

void emf_new(t_emf *emf, int nx[], t_fld box[], const float dt)
{
	int i;

	// Number of guard cells for linear interpolation
	int gc[2][2] = {{1, 2}, {1, 2}};

	// Allocate global arrays
	emf->total_size = (gc[0][0] + nx[0] + gc[0][1]) * (gc[1][0] + nx[1] + gc[1][1]);

	emf->E_buf = calloc(emf->total_size, sizeof(t_vfld));
	emf->B_buf = calloc(emf->total_size, sizeof(t_vfld));

	assert(emf->E_buf && emf->B_buf);

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
	emf->moving_window = 0;
	emf->n_move = 0;
}

void emf_delete(t_emf *emf)
{
	free(emf->E_buf);
	free(emf->B_buf);

	emf->E_buf = NULL;
	emf->B_buf = NULL;

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

	return sqrt(sqrt(rWl2)) * exp(-rho2 * rWl2 / (laser->W0 * laser->W0)) * cos(laser->omega0 * (z + curv) - gouy_shift);
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

void div_corr_x(t_emf *emf)
{
	int i, j;

	double ex, bx;

	t_vfld *restrict E = emf->E;
	t_vfld *restrict B = emf->B;
	const int nrow = emf->nrow;
	const double dx_dy = emf->dx[0] / emf->dx[1];

	for (j = 0; j < emf->nx[1]; j++)
	{
		ex = 0.0;
		bx = 0.0;
		for (i = emf->nx[0] - 1; i >= 0; i--)
		{
			ex += dx_dy * (E[i + 1 + j * nrow].y - E[i + 1 + (j - 1) * nrow].y);
			E[i + j * nrow].x = ex;

			bx += dx_dy * (B[i + (j + 1) * nrow].y - B[i + j * nrow].y);
			B[i + j * nrow].x = bx;
		}

	}
}

void emf_add_laser(t_emf *const emf, t_emf_laser *laser)
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
	int i, j, nrow;

	t_fld r_center, z, z_2, r, r_2;
	t_fld amp, lenv, lenv_2, k;
	t_fld dx, dy;
	t_fld cos_pol, sin_pol;

	t_vfld *restrict E = emf->E;
	t_vfld *restrict B = emf->B;

	nrow = emf->nrow;
	dx = emf->dx[0];
	dy = emf->dx[1];

	r_center = laser->axis;
	amp = laser->omega0 * laser->a0;

	cos_pol = cos(laser->polarization);
	sin_pol = sin(laser->polarization);

	switch (laser->type)
	{
		case PLANE:
			k = laser->omega0;

			for (i = 0; i < emf->nx[0]; i++)
			{
				z = i * dx;
				z_2 = z + dx / 2;

				lenv = amp * lon_env(laser, z);
				lenv_2 = amp * lon_env(laser, z_2);

				for (j = 0; j < emf->nx[1]; j++)
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

			for (i = 0; i < emf->nx[0]; i++)
			{
				z = i * dx;
				z_2 = z + dx / 2;

				lenv = amp * lon_env(laser, z);
				lenv_2 = amp * lon_env(laser, z_2);

				for (j = 0; j < emf->nx[1]; j++)
				{
					r = j * dy - r_center;
					r_2 = r + dy / 2;

					// E[i + j*nrow].x += 0.0
					E[i + j * nrow].y += +lenv * gauss_phase(laser, z, r_2) * cos_pol;
					E[i + j * nrow].z += +lenv * gauss_phase(laser, z, r) * sin_pol;

					// B[i + j*nrow].x += 0.0
					B[i + j * nrow].y += -lenv_2 * gauss_phase(laser, z_2, r) * sin_pol;
					B[i + j * nrow].z += +lenv_2 * gauss_phase(laser, z_2, r_2) * cos_pol;
				}
			}
			div_corr_x(emf);

			break;
		default:
			break;
	}

	// Set guard cell values
	emf_update_gc(emf->E_buf, emf->B_buf, emf->nrow, emf->nx, emf->gc, emf->total_size, emf->moving_window);
	#pragma omp taskwait
}

/*********************************************************************************************
 Diagnostics
 *********************************************************************************************/

void emf_report(const t_emf *emf, const char field, const char fc, const char path[64])
{
	int i, j;
	char vfname[3];

	// Choose field to save
	t_vfld *restrict f;
	switch (field)
	{
		case EFLD:
			f = emf->E;
			vfname[0] = 'E';
			break;
		case BFLD:
			f = emf->B;
			vfname[0] = 'B';
			break;
		default:
			fprintf(stderr, "Invalid field type selected, returning\n");
			return;
	}

	// Pack the information
	float *restrict const buf = malloc(emf->nx[0] * emf->nx[1] * sizeof(float));
	float *restrict p = buf;
	switch (fc)
	{
		case 0:
			for (j = 0; j < emf->nx[1]; j++)
			{
				for (i = 0; i < emf->nx[0]; i++)
				{
					p[i] = f[i].x;
				}
				p += emf->nx[0];
				f += emf->nrow;
			}
			vfname[1] = '1';
			break;
		case 1:
			for (j = 0; j < emf->nx[1]; j++)
			{
				for (i = 0; i < emf->nx[0]; i++)
				{
					p[i] = f[i].y;
				}
				p += emf->nx[0];
				f += emf->nrow;
			}
			vfname[1] = '2';
			break;
		case 2:
			for (j = 0; j < emf->nx[1]; j++)
			{
				for (i = 0; i < emf->nx[0]; i++)
				{
					p[i] = f[i].z;
				}
				p += emf->nx[0];
				f += emf->nrow;
			}
			vfname[1] = '3';
			break;
		default:
			fprintf(stderr, "Invalid field component selected, returning\n");
			return;
	}
	vfname[2] = 0;

	t_zdf_grid_axis axis[2];
	axis[0] = (t_zdf_grid_axis) {.min = 0.0, .max = emf->box[0], .label = "x_1", .units = "c/\\omega_p"};

	axis[1] = (t_zdf_grid_axis) {.min = 0.0, .max = emf->box[1], .label = "x_2", .units = "c/\\omega_p"};

	t_zdf_grid_info info = {.ndims = 2, .label = vfname, .units = "m_e c \\omega_p e^{-1}", .axis = axis};

	info.nx[0] = emf->nx[0];
	info.nx[1] = emf->nx[1];

	t_zdf_iteration iter = {.n = emf->iter, .t = emf->iter * emf->dt, .time_units = "1/\\omega_p"};

	zdf_save_grid(buf, &info, &iter, path);

	// free local data
	free(buf);

}

double emf_get_energy(t_emf *emf)
{
	t_vfld *const restrict E = emf->E;
	t_vfld *const restrict B = emf->B;
	double result = 0;

	for(unsigned int i = 0; i < emf->nx[0] * emf->nx[1]; i++)
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

void emf_report_magnitude(const t_emf *emf, const char name[64])
{
	char filenameE[128];
	char filenameB[128];

	t_fld *restrict E_magnitude = malloc(emf->nx[0] * emf->nx[1] * sizeof(t_fld));
	t_fld *restrict B_magnitude = malloc(emf->nx[0] * emf->nx[1] * sizeof(t_fld));

	const unsigned int nrows = emf->nrow;
	t_vfld *const restrict E = emf->E;
	t_vfld *const restrict B = emf->B;

	for (unsigned int j = 0; j < emf->nx[1]; j++)
	{
		for (unsigned int i = 0; i < emf->nx[0]; i++)
		{
			E_magnitude[i + j * emf->nx[0]] = sqrt(E[i + j * nrows].x * E[i + j * nrows].x + E[i + j * nrows].y * E[i + j * nrows].y
							+ E[i + j * nrows].z * E[i + j * nrows].z);

			B_magnitude[i + j * emf->nx[0]] = sqrt(B[i + j * nrows].x * B[i + j * nrows].x + B[i + j * nrows].y * B[i + j * nrows].y
							+ B[i + j * nrows].z * B[i + j * nrows].z);
		}
	}

	sprintf(filenameE, "e_mag_map_%d.csv", emf->iter);
	sprintf(filenameB, "b_mag_map_%d.csv", emf->iter);

	save_data_csv(E_magnitude, emf->nx[0], emf->nx[1], filenameE, name);
	save_data_csv(B_magnitude, emf->nx[0], emf->nx[1], filenameB, name);

	free(E_magnitude);
	free(B_magnitude);
}

/*********************************************************************************************
 Field solver
 *********************************************************************************************/

#pragma omp task inout(E_buf[0; total_size]) inout(B_buf[0; total_size])
void yee_b(t_vfld *restrict B_buf, const t_vfld *restrict E_buf, const t_fld dt_dx,
        const t_fld dt_dy, const int nrow, const int nx[2], const int total_size)
{
	// Canonical implementation
	#pragma omp for schedule(static)
	for (int j = 0; j <= nx[1] + 1; j++)
	{
		for (int i = 0; i <= nx[0] + 1; i++)
		{
			B_buf[i + j * nrow].x += (-dt_dy * (E_buf[i + (j + 1) * nrow].z - E_buf[i + j * nrow].z));
			B_buf[i + j * nrow].y += (dt_dx * (E_buf[(i + 1) + j * nrow].z - E_buf[i + j * nrow].z));
			B_buf[i + j * nrow].z += (-dt_dx * (E_buf[(i + 1) + j * nrow].y - E_buf[i + j * nrow].y)
					+ dt_dy * (E_buf[i + (j + 1) * nrow].x - E_buf[i + j * nrow].x));
		}
	}
}

#pragma omp task inout(E_buf[0; total_size]) inout(B_buf[0; total_size]) in(J_buf[0; total_size])
void yee_e(const t_vfld *restrict B_buf, t_vfld *restrict E_buf, const t_vfld *restrict J_buf,
        const const t_fld dt_dx, const t_fld dt_dy, const float dt, const int nrow_e,
        const int nrow_j, const int nx[2], const int total_size)
{
	// Canonical implementation
	#pragma omp for schedule(static)
	for (int j = 1; j <= nx[1] + 2; j++)
	{
		for (int i = 1; i <= nx[0] + 2; i++)
		{
			E_buf[i + j * nrow_e].x += (+dt_dy * (B_buf[i + j * nrow_e].z - B_buf[i + (j - 1) * nrow_e].z))
					- dt * J_buf[i + j * nrow_j].x;
			E_buf[i + j * nrow_e].y += (-dt_dx * (B_buf[i + j * nrow_e].z - B_buf[(i - 1) + j * nrow_e].z))
					- dt * J_buf[i + j * nrow_j].y;
			E_buf[i + j * nrow_e].z += (+dt_dx * (B_buf[i + j * nrow_e].y - B_buf[(i - 1) + j * nrow_e].y)
					- dt_dy * (B_buf[i + j * nrow_e].x - B_buf[i + (j - 1) * nrow_e].x)) - dt * J_buf[i + j * nrow_j].z;
		}
	}
}

// This code operates with periodic boundaries
#pragma omp task inout(E_buf[0; total_size]) inout(B_buf[0; total_size])
void emf_update_gc(t_vfld *restrict E_buf, t_vfld *restrict B_buf, const int nrow, const int nx[2],
        const int gc[2][2], const int total_size, const int moving_window)
{
	// For moving window don't update x boundaries
	if (!moving_window)
	{
		// x
		#pragma omp for schedule(static)
		for (int j = 0; j < gc[1][0] + nx[1] + gc[1][1]; j++)
		{
			// lower
			for (int i = 0; i < gc[0][0]; i++)
			{
				E_buf[i + j * nrow].x = E_buf[nx[0] + i + j * nrow].x;
				E_buf[i + j * nrow].y = E_buf[nx[0] + i + j * nrow].y;
				E_buf[i + j * nrow].z = E_buf[nx[0] + i + j * nrow].z;

				B_buf[i + j * nrow].x = B_buf[nx[0] + i + j * nrow].x;
				B_buf[i + j * nrow].y = B_buf[nx[0] + i + j * nrow].y;
				B_buf[i + j * nrow].z = B_buf[nx[0] + i + j * nrow].z;
			}

			// upper
			for (int i = gc[0][0]; i < gc[0][0] + gc[0][1]; i++)
			{
				E_buf[nx[0] + i + j * nrow].x = E_buf[i + j * nrow].x;
				E_buf[nx[0] + i + j * nrow].y = E_buf[i + j * nrow].y;
				E_buf[nx[0] + i + j * nrow].z = E_buf[i + j * nrow].z;

				B_buf[nx[0] + i + j * nrow].x = B_buf[i + j * nrow].x;
				B_buf[nx[0] + i + j * nrow].y = B_buf[i + j * nrow].y;
				B_buf[nx[0] + i + j * nrow].z = B_buf[i + j * nrow].z;
			}
		}
	}

	// y
	#pragma omp for schedule(static)
	for (int i = 0; i < nrow; i++)
	{
		// lower
		for (int j = 0; j < gc[1][0]; j++)
		{
			E_buf[i + j * nrow].x = E_buf[i + (nx[1] + j) * nrow].x;
			E_buf[i + j * nrow].y = E_buf[i + (nx[1] + j) * nrow].y;
			E_buf[i + j * nrow].z = E_buf[i + (nx[1] + j) * nrow].z;

			B_buf[i + j * nrow].x = B_buf[i + (nx[1] + j) * nrow].x;
			B_buf[i + j * nrow].y = B_buf[i + (nx[1] + j) * nrow].y;
			B_buf[i + j * nrow].z = B_buf[i + (nx[1] + j) * nrow].z;
		}

		// upper
		for (int j = gc[1][0]; j < gc[1][0] + gc[1][1]; j++)
		{
			E_buf[i + (nx[1] + j) * nrow].x = E_buf[i + j * nrow].x;
			E_buf[i + (nx[1] + j) * nrow].y = E_buf[i + j * nrow].y;
			E_buf[i + (nx[1] + j) * nrow].z = E_buf[i + j * nrow].z;

			B_buf[i + (nx[1] + j) * nrow].x = B_buf[i + j * nrow].x;
			B_buf[i + (nx[1] + j) * nrow].y = B_buf[i + j * nrow].y;
			B_buf[i + (nx[1] + j) * nrow].z = B_buf[i + j * nrow].z;
		}
	}
}

#pragma omp task inout(E_buf[0; total_size]) inout(B_buf[0; total_size])
void emf_move_window(t_vfld *restrict E_buf, t_vfld *restrict B_buf, const int nrow, const int nx[2],
		const int gc[2][2], const int total_size)
{
	const t_vfld zero_fld = {0., 0., 0.};

	// Shift data left 1 cell and zero rightmost cells
	#pragma omp for schedule(static)
	for (int j = 0; j < gc[1][0] + nx[1] + gc[1][1]; j++)
	{
		for (int i = 0; i < nrow - 1; i++)
		{
			E_buf[i + j * nrow] = E_buf[i + j * nrow + 1];
			B_buf[i + j * nrow] = B_buf[i + j * nrow + 1];
		}

		for (int i = gc[0][0] + nx[0] - 1; i < nrow; i++)
		{
			E_buf[i + j * nrow] = zero_fld;
			B_buf[i + j * nrow] = zero_fld;
		}
	}
}

// Perform the local integration of the fields (and post processing). OpenAcc Task
void emf_advance(t_emf *emf, const t_current *current)
{
	const t_fld dt = emf->dt;
	const t_fld dt_dx = dt / emf->dx[0];
	const t_fld dt_dy = dt / emf->dx[1];

	// Advance EM field using Yee algorithm modified for having E and B time centered
	yee_b(emf->B_buf, emf->E_buf, dt_dx / 2.0f, dt_dy / 2.0f, emf->nrow, emf->nx, emf->total_size);
	yee_e(emf->B_buf, emf->E_buf, current->J_buf, dt_dx, dt_dy, dt, emf->nrow, current->nrow, emf->nx,
	        emf->total_size);
	yee_b(emf->B_buf, emf->E_buf, dt_dx / 2.0f, dt_dy / 2.0f, emf->nrow, emf->nx, emf->total_size);

	// Update guard cells with new values
	emf_update_gc(emf->E_buf, emf->B_buf, emf->nrow, emf->nx, emf->gc, emf->total_size, emf->moving_window);

	emf->iter++;

	if(emf->moving_window)
	{
		if ((emf->iter * emf->dt) > emf->dx[0] * (emf->n_move + 1))
		{
			emf_move_window(emf->E_buf, emf->B_buf, emf->nrow, emf->nx, emf->gc, emf->total_size);

			// Increase moving window counter
			emf->n_move++;
		}
	}
}
