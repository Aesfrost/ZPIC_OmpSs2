#include "zpic.h"
#include "particles.h"
#include "math.h"

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define LOCAL_BUFFER_SIZE 2048

/*********************************************************************************************
 Utilities
 *********************************************************************************************/

inline void atomicAdd_global_float(volatile __global float *address, float value)
{
	union {
		unsigned int uint32;
		float float32;
	} next, expected, current;
	
	current.float32 = *address;
	
	do
	{
		expected.float32 = current.float32;
		next.float32 = expected.float32 + value;
		current.uint32 = atomic_cmpxchg((volatile __global unsigned int *) address, expected.uint32, next.uint32);
	} while(current.uint32 != expected.uint32);
}


/*********************************************************************************************
 Particle Advance
 *********************************************************************************************/
// EM fields interpolation. OpenAcc Task
void interpolate_fld_opencl(__global const t_vfld *restrict const E, __global const t_vfld *restrict const B,
							 const int nrow, const int ix, const int iy, const t_fld x, const t_fld y,
							 __local t_vfld *restrict const Ep, __local t_vfld *restrict const Bp)
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
	+ (B[ih + (jh + 1) * nrow].z * (1.0f - w1h) + B[ih + 1 + (jh + 1) * nrow].z * w1h)
	* w2h;
}

// Current deposition (adapted Villasenor-Bunemann method). OpenAcc task
void dep_current_opencl(int ix, int iy, int di, int dj, float x0, float y0, float dx, float dy,
						float qnx, float qny, float qvz, __global t_vfld *restrict const J, const int nrow)
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
	for (int k = 0; k < vnp; k++)
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
		
		
		atomicAdd_global_float(&J[vp[k].ix + nrow * vp[k].iy].x, wl1 * wp1[0]);
		atomicAdd_global_float(&J[vp[k].ix + nrow * (vp[k].iy + 1)].x, wl1 * wp1[1]);
		
		atomicAdd_global_float(&J[vp[k].ix + nrow * vp[k].iy].y, wl2 * wp2[0]);
		atomicAdd_global_float(&J[vp[k].ix + 1 + nrow * vp[k].iy].y, wl2 * wp2[1]);
		
		atomicAdd_global_float(&J[vp[k].ix + nrow * vp[k].iy].z, vp[k].qvz
		* (S0x[0] * S0y[0] + S1x[0] * S1y[0] + (S0x[0] * S1y[0] - S1x[0] * S0y[0]) / 2.0f));
		atomicAdd_global_float(&J[vp[k].ix + 1 + nrow * vp[k].iy].z, vp[k].qvz
		* (S0x[1] * S0y[0] + S1x[1] * S1y[0] + (S0x[1] * S1y[0] - S1x[1] * S0y[0]) / 2.0f));
		atomicAdd_global_float(&J[vp[k].ix + nrow * (vp[k].iy + 1)].z, vp[k].qvz
		* (S0x[0] * S0y[1] + S1x[0] * S1y[1] + (S0x[0] * S1y[1] - S1x[0] * S0y[1]) / 2.0f));
		atomicAdd_global_float(&J[vp[k].ix + 1 + nrow * (vp[k].iy + 1)].z, vp[k].qvz
		* (S0x[1] * S0y[1] + S1x[1] * S1y[1] + (S0x[1] * S1y[1] - S1x[1] * S0y[1]) / 2.0f));
	}
}

// Particle advance (OpenAcc)
__kernel void spec_advance_openacc(__global t_particle_vector *restrict particle_vector, __global const t_vfld *restrict E, 
						  __global const t_vfld *restrict B, __global t_vfld *restrict const J, 
						  const int nrow, const int limits_y[2], const t_part_data tem,
						  const t_part_data dt_dx, const t_part_data dt_dy, const t_part_data qnx, 
						  const t_part_data qny)
{
	const int id = get_global_id(0);
	const int stride = get_global_size(0);
	
	// Advance particles
	for (int k = id; k < particle_vector->size; k += stride)
	{
		if(!particle_vector->safe_to_delete[k])
		{
			t_vfld Ep, Bp;
			
			// Interpolate fields
			interpolate_fld_opencl(E, B, nrow, particle_vector->ix[k], particle_vector->iy[k] - limits_y[0], particle_vector->x[k], particle_vector->y[k], &Ep, &Bp);
			
			// Advance u using Boris scheme
			Ep.x *= tem;
			Ep.y *= tem;
			Ep.z *= tem;
			
			t_part_data utx = particle_vector->ux[k] + Ep.x;
			t_part_data uty = particle_vector->uy[k] + Ep.y;
			t_part_data utz = particle_vector->uz[k] + Ep.z;
			
			// Perform first half of the rotation
			t_part_data utsq = utx * utx + uty * uty + utz * utz;
			t_part_data gtem = tem / sqrtf(1.0f + utsq);
			
			Bp.x *= gtem;
			Bp.y *= gtem;
			Bp.z *= gtem;
			
			particle_vector->ux[k] = utx + uty * Bp.z - utz * Bp.y;
			particle_vector->uy[k] = uty + utz * Bp.x - utx * Bp.z;
			particle_vector->uz[k] = utz + utx * Bp.y - uty * Bp.x;
			
			// Perform second half of the rotation
			t_part_data Bp_mag = Bp.x * Bp.x + Bp.y * Bp.y + Bp.z * Bp.z;
			t_part_data otsq = 2.0f / (1.0f + Bp_mag);
			
			Bp.x *= otsq;
			Bp.y *= otsq;
			Bp.z *= otsq;
			
			utx += particle_vector->uy[k] * Bp.z - particle_vector->uz[k] * Bp.y;
			uty += particle_vector->uz[k] * Bp.x - particle_vector->ux[k] * Bp.z;
			utz += particle_vector->ux[k] * Bp.y - particle_vector->uy[k] * Bp.x;
			
			// Perform second half of electric field acceleration
			particle_vector->ux[k] = utx + Ep.x;
			particle_vector->uy[k] = uty + Ep.y;
			particle_vector->uz[k] = utz + Ep.z;
			
			// Push particle
			t_part_data usq = particle_vector->ux[k] * particle_vector->ux[k]
			+ particle_vector->uy[k] * particle_vector->uy[k]
			+ particle_vector->uz[k] * particle_vector->uz[k];
			t_part_data rg = 1.0f / sqrtf(1.0f + usq);
			
			t_part_data dx = dt_dx * rg * particle_vector->ux[k];
			t_part_data dy = dt_dy * rg * particle_vector->uy[k];
			
			t_part_data x1 = particle_vector->x[k] + dx;
			t_part_data y1 = particle_vector->y[k] + dy;
			
			int di = (x1 >= 1.0f) - (x1 < 0.0f);
			int dj = (y1 >= 1.0f) - (y1 < 0.0f);
			
			t_part_data qvz = spec->q * particle_vector->uz[k] * rg;
			
			dep_current_opencl(particle_vector->ix[k], particle_vector->iy[k] - limits_y[0], di, dj,
								particle_vector->x[k], particle_vector->y[k], dx, dy, qnx, qny, qvz, J, nrow);
			
			// Store results
			particle_vector->x[k] = x1 - di;
			particle_vector->y[k] = y1 - dj;
			particle_vector->ix[k] += di;
			particle_vector->iy[k] += dj;
		}
	}
}

/*********************************************************************************************
 Post Processing 1 (Region Check)                                                              
 *********************************************************************************************/

// Transfer particles between regions (if applicable). OpenAcc Task
__kernel void spec_post_processing_1_openacc(__global t_particle_vector *restrict particle_vector, __global t_particle_vector *restrict upper_buffer,
									__global t_particle_vector *restrict lower_buffer, const int limits_y[2], const int nx[2], 
									const bool shift)
{
	const int id = get_global_id(0);
	const int stride = get_global_size(0);
	
	int iy, idx;

	for (int i = id; i < particles_vector->size; i += stride)
	{
		if(!particles_vector->safe_to_delete[i])
		{
			
			if (spec->moving_window)
			{
				// Shift particles left
				if (shift) particles_vector->ix[i]--;
				
				// Verify if the particle is leaving the region
				if ((particles_vector->ix[i] >= 0) && (particles_vector->ix[i] < nx[0]))
				{
					iy = particles_vector->iy[i];
					
					if (iy < limits_y[0])
					{
						if (iy < 0) particles_vector->iy[i] += nx[1];
						
						// Reserve a position in the vector
						idx = atomic_inc(&lower_buffer->size);
						
						lower_buffer->ix[idx] = particles_vector->ix[i];
						lower_buffer->iy[idx] = particles_vector->iy[i];
						lower_buffer->x[idx] = particles_vector->x[i];
						lower_buffer->y[idx] = particles_vector->y[i];
						lower_buffer->ux[idx] = particles_vector->ux[i];
						lower_buffer->uy[idx] = particles_vector->uy[i];
						lower_buffer->uz[idx] = particles_vector->uz[i];
						lower_buffer->safe_to_delete[idx] = false;
						
						particles_vector->safe_to_delete[i] = true; // Mark the particle as invalid
						
					} else if (iy >= limits_y[1])
					{
						if (iy >= nx[1]) particles_vector->iy[i] -= nx[1];
						
						// Reserve a position in the vector
						idx = atomic_inc(&upper_buffer->size);
						
						upper_buffer->ix[idx] = particles_vector->ix[i];
						upper_buffer->iy[idx] = particles_vector->iy[i];
						upper_buffer->x[idx] = particles_vector->x[i];
						upper_buffer->y[idx] = particles_vector->y[i];
						upper_buffer->ux[idx] = particles_vector->ux[i];
						upper_buffer->uy[idx] = particles_vector->uy[i];
						upper_buffer->uz[idx] = particles_vector->uz[i];
						upper_buffer->safe_to_delete[idx] = false;
						
						particles_vector->safe_to_delete[i] = true; // Mark the particle as invalid
						
					}
				} else particles_vector->safe_to_delete[i] = true; // Mark the particle as invalid
			} else
			{
				//Periodic boundaries for both axis
				if (particles_vector->ix[i] < 0) particles_vector->ix[i] += nx[0];
				else if (particles_vector->ix[i] >= nx[0]) particles_vector->ix[i] -= nx[0];
				
				iy = particles_vector->iy[i];
				
				// Check if the particle is leaving the box
				if (iy < limits_y[0])
				{
					if (iy < 0) particles_vector->iy[i] += nx[1];
					
					// Reserve a position in the vector
					idx = atomic_inc(&lower_buffer->size);
					
					lower_buffer->ix[idx] = particles_vector->ix[i];
					lower_buffer->iy[idx] = particles_vector->iy[i];
					lower_buffer->x[idx] = particles_vector->x[i];
					lower_buffer->y[idx] = particles_vector->y[i];
					lower_buffer->ux[idx] = particles_vector->ux[i];
					lower_buffer->uy[idx] = particles_vector->uy[i];
					lower_buffer->uz[idx] = particles_vector->uz[i];
					lower_buffer->safe_to_delete[idx] = false;
					
					particles_vector->safe_to_delete[i] = true; // Mark the particle as invalid
					
				} else if (iy >= limits_y[1])
				{
					if (iy >= nx[1]) particles_vector->iy[i] -= nx[1];
					
					idx = atomic_inc(&upper_buffer->size);
					
					upper_buffer->ix[idx] = particles_vector->ix[i];
					upper_buffer->iy[idx] = particles_vector->iy[i];
					upper_buffer->x[idx] = particles_vector->x[i];
					upper_buffer->y[idx] = particles_vector->y[i];
					upper_buffer->ux[idx] = particles_vector->ux[i];
					upper_buffer->uy[idx] = particles_vector->uy[i];
					upper_buffer->uz[idx] = particles_vector->uz[i];
					upper_buffer->safe_to_delete[idx] = false;
					
					particles_vector->safe_to_delete[i] = true; // Mark the particle as invalid
				} 
			}
		}
	}
}

/*********************************************************************************************
 Prefix Sum                                                                             
 *********************************************************************************************/
__kernel void prefix_sum_local(__global int *restrict vector, __global int *restrict block_sum,
							   const unsigned int global_size, const unsigned int num_groups)
{
	__local int local_buffer[LOCAL_BUFFER_SIZE];
	
	const int id = get_local_id(0);
	const int group_id = get_group_id(0);
	const int begin_idx = group_id * LOCAL_BUFFER_SIZE;
	int offset = 1;
	const int idx = 2 * id + begin_idx;
	
	if(idx + 1 < global_size)
	{
		local_buffer[2 * id] = vector[idx];
		local_buffer[2 * id + 1] = vector[idx + 1];
	}else 
	{
		local_buffer[2 * id] = 0;
		local_buffer[2 * id + 1] = 0;
	}
	
	for(int d = LOCAL_BUFFER_SIZE >> 1; d > 0;  d >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(id < d)
		{
			int idx1 = offset * (2 * id + 1) - 1;
			int idx2 = offset * (2 * id + 2) - 1;
			
			local_buffer[idx2] += local_buffer[idx1];
		}
		
		offset *= 2;
	}
	
	if(!id)
	{
		if(num_groups > 1) block_sum[group_id] = local_buffer[LOCAL_BUFFER_SIZE - 1];
		local_buffer[LOCAL_BUFFER_SIZE - 1] = 0;
	}
	
	for(int d = 1; d < LOCAL_BUFFER_SIZE; d *= 2)
	{
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(id < d)
		{
			int idx1 = offset * (2 * id + 1) - 1;
			int idx2 = offset * (2 * id + 2) - 1;
			
			int temp = local_buffer[idx1];
			local_buffer[idx1] = local_buffer[idx2];
			local_buffer[idx2] += temp;
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(idx + 1 < global_size)
	{
		vector[idx] = local_buffer[2 * id];
		vector[idx + 1] = local_buffer[2 * id + 1];	
	}
}

__kernel void add_group_sum(__global int *restrict vector, __global int *restrict block_sum, 
							const unsigned int global_size, const unsigned int num_groups)
{
	const int group_id = get_group_id(0);
	const int id = get_global_id(0);
	const int idx = 2 * id;
	
	if(idx < global_size && group_id > 0)
	{
		vector[idx] += block_sum[group_id];
		vector[idx + 1] += block_sum[group_id];
	}	
}

/*********************************************************************************************
 Sort                                                                                    
 *********************************************************************************************/

__kernel void spec_bin_count (__global const t_particle_vector *restrict particle_vector, __global int *restrict count, 
								  const int n_bins->x)
{
	const int id = get_global_id(0);
	const int stride = get_global_size(0);
	
	int2 idx;
	
	for(int i = id; i < particle_vector->size; i += stride)
	{
		if(!particle_vector->safe_to_delete[i])
		{
			idx.x = particle_vector->ix[i] / BIN_SIZE;
			idx.y = particle_vector->iy[i] / BIN_SIZE;
			
			atomic_inc(&count[idx.x + idx.y * n_bins_x]);
		}
	}
}

__kernel void spec_bin_sort (__global const t_particle_vector *restrict particle_vector, __global t_particle_vector *restrict bins, 
							 __global int *restrict bin_idx, const int n_bins->x)
{
	const int id = get_global_id(0);
	const int stride = get_global_size(0);
	
	int2 bin_pos;
	int idx;
	
	for(int i = id; i < particle_vector->size; i += stride)
	{
		if(!particles_vector->safe_to_delete[i])
		{
			bin_pos.x = particle_vector->ix[i] / BIN_SIZE;
			bin_pos.y = particle_vector->iy[i] / BIN_SIZE;
			
			idx = atomic_inc(&bin_idx[bin_pos.x + bin_pos.y * n_bins_x])
			
			bins->ix[idx] = particles_vector->ix[i];
			bins->iy[idx] = particles_vector->iy[i];
			bins->x[idx] = particles_vector->x[i];
			bins->y[idx] = particles_vector->y[i];
			bins->ux[idx] = particles_vector->ux[i];
			bins->uy[idx] = particles_vector->uy[i];
			bins->uz[idx] = particles_vector->uz[i];
		}
	}
}

__kernel void spec_memcopy_bins (__global const t_particle_vector *restrict particle_vector, __global t_particle_vector *restrict bins, 
								 const int size)
{
	const int id = get_global_id(0);
	const int stride = get_global_size(0);
	
	for(int i = id; i < size; i += stride)
	{
		particle_vector->ix[i] = bins->ix[i];
		particle_vector->iy[i] = bins->iy[i];
		particle_vector->x[i] = bins->x[i];
		particle_vector->y[i] = bins->y[i];
		particle_vector->ux[i] = bins->ux[i];
		particle_vector->uy[i] = bins->uy[i];
		particle_vector->uz[i] = bins->uz[i];
		particle_vector->safe_to_delete[i] = false;
	}
}

