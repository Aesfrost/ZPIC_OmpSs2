#include "current.h"
#include <stdlib.h>

void kernel_x_openacc(t_current *const current, const t_fld sa, const t_fld sb)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;
	t_vfld *restrict J_buf = malloc(nrow * current->nx[1] * sizeof(t_vfld));
	t_vfld *restrict J_aux = J_buf + current->gc[0][0];

	#pragma acc parallel loop independent collapse(2)
	for (int j = 0; j < current->nx[1]; ++j)
		for (int i = 0; i < nrow; ++i)
			J_buf[i + j * nrow] = J[i - 1 + j * nrow];

	#pragma acc parallel loop collapse(2) independent
	for (int j = 0; j < current->nx[1]; ++j)
	{
		for (int i = 0; i < current->nx[0]; ++i)
		{
			J[i + j * nrow].x = J_aux[i - 1 + j * nrow].x * sa + J_aux[i + j * nrow].x * sb
					+ J_aux[i + 1 + j * nrow].x * sa;
			J[i + j * nrow].y = J_aux[i - 1 + j * nrow].y * sa + J_aux[i + j * nrow].y * sb
					+ J_aux[i + 1 + j * nrow].y * sa;
			J[i + j * nrow].z = J_aux[i - 1 + j * nrow].z * sa + J_aux[i + j * nrow].z * sb
					+ J_aux[i + 1 + j * nrow].z * sa;
		}
	}

	if(!current->moving_window)
	{
		#pragma acc parallel loop independent collapse(2)
		for (int j = 0; j < current->nx[1]; ++j)
		{
			for (int i = -current->gc[0][0]; i < current->gc[0][1]; i++)
				if(i < 0) J[i + j * nrow] = J[current->nx[0] + i + j * nrow];
				else J[current->nx[0] + i + j * nrow] = J[i + j * nrow];
		}
	}

	free(J_buf);
}

void current_smooth_openacc(t_current *const current)
{
	// filter kernel [sa, sb, sa]
	t_fld sa, sb;

	// x-direction filtering
	if (current->smooth.xtype != NONE)
	{
		// binomial filter
		sa = 0.25;
		sb = 0.5;
		for (int i = 0; i < current->smooth.xlevel; i++)
			kernel_x_openacc(current, 0.25, 0.5);

		// Compensator
		if (current->smooth.xtype == COMPENSATED)
		{
			get_smooth_comp(current->smooth.xlevel, &sa, &sb);
			kernel_x_openacc(current, sa, sb);
		}
	}

	// y-direction filtering
	if (current->smooth.ytype != NONE)
	{
		// binomial filter
		sa = 0.25;
		sb = 0.5;
		for (int i = 0; i < current->smooth.xlevel; i++)
			kernel_y(current, 0.25, 0.5);

		// Compensator
		if (current->smooth.ytype == COMPENSATED)
		{
			get_smooth_comp(current->smooth.ylevel, &sa, &sb);
			kernel_y(current, sa, sb);
		}
	}
}

void current_update_openacc(t_current *current)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;

	// x
	if (!current->moving_window)
	{
		#pragma acc parallel loop independent collapse(2)
		for (int j = -current->gc[1][0]; j < current->nx[1] + current->gc[1][1]; j++)
		{
			// lower - add the values from upper boundary (both gc and inside box)
			for (int i = -current->gc[0][0]; i < current->gc[0][1]; i++)
			{
				J[i + j * nrow].x += J[current->nx[0] + i + j * nrow].x;
				J[i + j * nrow].y += J[current->nx[0] + i + j * nrow].y;
				J[i + j * nrow].z += J[current->nx[0] + i + j * nrow].z;

				J[current->nx[0] + i + j * nrow] = J[i + j * nrow];
			}
		}
	}

	// y
	#pragma acc parallel loop independent collapse(2)
	for (int i = -current->gc[0][0]; i < current->nx[0] + current->gc[0][1]; i++)
	{
		// lower - add the values from upper boundary (both gc and inside box)
		for (int j = -current->gc[1][0]; j < current->gc[1][1]; j++)
		{
			J[i + j * nrow].x += J[i + (current->nx[1] + j) * nrow].x;
			J[i + j * nrow].y += J[i + (current->nx[1] + j) * nrow].y;
			J[i + j * nrow].z += J[i + (current->nx[1] + j) * nrow].z;

			J[i + (current->nx[1] + j) * nrow] = J[i + j * nrow];
		}
	}

	// Smoothing
	current_smooth_openacc(current);

	current->iter++;
}
