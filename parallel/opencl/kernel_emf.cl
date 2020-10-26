#include "emf.h"

__kernel void yee_b_opencl(__global float3 *restrict B, __global const float3 *restrict E, const int nrow, 
				  const int nx[2], const float dt_dx, const float dt_dy)
{

	const int2 id = (int2) (get_global_id(0), get_global_id(1));
	const int2 stride = (int2) (get_global_size(0), get_global_size(1));
	
	for (int j = id.x - 1; j <= nx[1]; j += stride.x)
	{
		for (int i = id.y - 1; i <= nx[0]; i += stride.y)
		{
			B[i + j * nrow].x += (-dt_dy * (E[i + (j + 1) * nrow].z - E[i + j * nrow].z));
			B[i + j * nrow].y += (dt_dx * (E[(i + 1) + j * nrow].z - E[i + j * nrow].z));
			B[i + j * nrow].z += (-dt_dx * (E[(i + 1) + j * nrow].y - E[i + j * nrow].y)
			+ dt_dy * (E[i + (j + 1) * nrow].x - E[i + j * nrow].x));
		}
	}
}

__kernel void yee_e_opencl(__global const float3 *restrict B, __global float3 *restrict E, 
						   __global const float3 *restrict J, const int nrow_e, const int nrow_j, 
						   const int nx[2], const float dt_dx, const float dt_dy)
{
	
	const int2 id = (int2) (get_global_id(0), get_global_id(1));
	const int2 stride = (int2) (get_global_size(0), get_global_size(1));
	
	for (int j = id.x; j <= emf->nx[1] + 1; j += stride.y)
	{
		for (int i = idx.y; i <= emf->nx[0] + 1; i += stride.x)
		{
			E[i + j * nrow_e].x += (+dt_dy * (B[i + j * nrow_e].z - B[i + (j - 1) * nrow_e].z))
			- dt * J[i + j * nrow_j].x;
			E[i + j * nrow_e].y += (-dt_dx * (B[i + j * nrow_e].z - B[(i - 1) + j * nrow_e].z))
			- dt * J[i + j * nrow_j].y;
			E[i + j * nrow_e].z += (+dt_dx * (B[i + j * nrow_e].y - B[(i - 1) + j * nrow_e].y)
			- dt_dy * (B[i + j * nrow_e].x - B[i + (j - 1) * nrow_e].x)) - dt * J[i + j * nrow_j].z;
		}
	}
}
