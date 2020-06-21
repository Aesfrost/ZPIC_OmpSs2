/*********************************************************************************************
 ZPIC
 kernel_current.c

 Created by Nicolas Guidotti on 14/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#include "current.h"

// Each region is only responsible to do the reduction operation (y direction) in its top edge (OpenAcc)
void current_reduction_y_openacc(t_current *current)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;
	t_vfld *restrict const J_overlap = current->J_upper;

	#pragma acc parallel loop independent collapse(2)
	for (int j = -current->gc[1][0]; j < current->gc[1][1]; j++)
	{
		for (int i = -current->gc[0][0]; i < current->nx[0] + current->gc[0][1]; i++)
		{
			J[i + j * nrow].x += J_overlap[i + (j + current->gc[1][0]) * nrow].x;
			J[i + j * nrow].y += J_overlap[i + (j + current->gc[1][0]) * nrow].y;
			J[i + j * nrow].z += J_overlap[i + (j + current->gc[1][0]) * nrow].z;

			J_overlap[i + (j + current->gc[1][0]) * nrow] = J[i + j * nrow];
		}
	}
}

// Current reduction between ghost cells in the x direction (OpenAcc)
void current_reduction_x_openacc(t_current *current)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;
	t_vfld *restrict const J_overlap = &current->J[current->nx[0]];

	#pragma acc parallel loop independent collapse(2)
	for (int j = -current->gc[1][0]; j < current->nx[1] + current->gc[1][1]; j++)
	{
		for (int i = -current->gc[0][0]; i < current->gc[0][1]; i++)
		{
			J[i + j * nrow].x += J_overlap[i + j * nrow].x;
			J[i + j * nrow].y += J_overlap[i + j * nrow].y;
			J[i + j * nrow].z += J_overlap[i + j * nrow].z;

			J_overlap[i + j * nrow] = J[i + j * nrow];
		}
	}

	current->iter++;
}

// Update the ghost cells in the y direction (only the upper zone, OpenAcc)
void current_gc_update_y_openacc(t_current *current)
{
	const int nrow = current->nrow;
	t_vfld *restrict const J = current->J;
	t_vfld *restrict const J_overlap = current->J_upper;

	#pragma acc parallel loop independent collapse(2)
	for (int i = -current->gc[0][0]; i < current->nx[0] + current->gc[0][1]; i++)
	{
		for (int j = -current->gc[1][0]; j < current->gc[1][1]; j++)
		{
			if(j < 0)
			{
				J[i + j * nrow] = J_overlap[i + (j + current->gc[1][0]) * nrow];
			}else
			{
				J_overlap[i + (j + current->gc[1][0]) * nrow] = J[i + j * nrow];
			}
		}
	}
}

