#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#define TILE_SIZE 16
#define LTRIM(x) (x >= 1.0f) - (x < 0.0f)

typedef struct __attribute__((aligned(32)))
{
	float x0, x1, y0, y1, dx, dy;
	int ix, iy;
} t_vp;

typedef union {
	float3 fp3;
	float fp[3];
} float3_union;

/*********************************************************************************************
 Utilities
 *********************************************************************************************/
inline void atomic_add_fp32_global(volatile __global float *source, float value) 
{
	union {
		unsigned int u32;
		float f32;
	} next, expected, current;
	
	current.f32 = *source;
	
	do {
		expected.f32 = current.f32;
		next.f32 = expected.f32 + value;
		current.u32 = atomic_cmpxchg((volatile __global unsigned int*) source, expected.u32, next.u32);
	} while(current.u32 != expected.u32);
}

inline void atomic_add_float3_global(volatile __global float3_union *ptr, float3 value)
{
	if(value.x != 0) atomic_add_fp32_global(&ptr->fp[0], value.x);
	if(value.y != 0) atomic_add_fp32_global(&ptr->fp[1], value.y);
	if(value.z != 0) atomic_add_fp32_global(&ptr->fp[2], value.z);
}

inline void atomic_add_fp32_local(volatile __local float *source, float value) 
{
	union {
		unsigned int u32;
		float f32;
	} next, expected, current;
	
	current.f32 = *source;
	
	do {
		expected.f32 = current.f32;
		next.f32 = expected.f32 + value;
		current.u32 = atomic_cmpxchg((volatile __local unsigned int*) source, expected.u32, next.u32);
	} while(current.u32 != expected.u32);
}

inline void atomic_add_float3_local(volatile __local float3_union *ptr, float3 value)
{
	if(value.x != 0) atomic_add_fp32_local(&ptr->fp[0], value.x);
	if(value.y != 0) atomic_add_fp32_local(&ptr->fp[1], value.y);
	if(value.z != 0) atomic_add_fp32_local(&ptr->fp[2], value.z);
}

/*********************************************************************************************
 Spec Sort                                                                              
 **********************************************************************************************/

__kernel void spec_sort_1(__global const int2 *restrict part_cell_idx, __global const float2 *restrict part_positions, 
						  __global const float3 *restrict part_velocities, __global int2 *restrict temp_cell_idx, 
						  __global float2 *restrict temp_positions, __global float3 *restrict temp_velocities,
						  __global int *restrict target_idx, __global int *restrict counter, 
						  __global const int *restrict tile_offset,  __global const int *restrict temp_offset,
						  const int2 n_tiles, const int np, const int np_max, const int sort_size, const int nx0)
{
	const int local_id = get_local_id(0);
	const int stride = get_local_size(0);
	const int2 tile_idx = (int2) (get_group_id(0), get_group_id(1));
	
	const int current_tile = tile_idx.x + tile_idx.y * n_tiles.x;
	const int begin = tile_offset[current_tile];
	const int end = tile_offset[current_tile + 1];
	
	__local int offset;
	if(local_id == 0) offset = temp_offset[current_tile];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(int k = begin + local_id; k < end; k += stride)
	{
		int2 part_idx = part_cell_idx[k];
		int2 target_tile = part_idx / TILE_SIZE;

		if(part_idx.x < 0 || part_idx.x >= nx0 || k >= np)
		{
			int idx_t = atomic_inc(&offset);
			target_idx[idx_t] = k;
		}else if(any(target_tile != tile_idx))
	    {
			int idx_s = atomic_inc(&counter[target_tile.x + target_tile.y * n_tiles.x]);
			int idx_t = atomic_inc(&offset);
			
			temp_cell_idx[idx_s] = part_idx;
			temp_velocities[idx_s] = part_velocities[k];
			temp_positions[idx_s] = part_positions[k];
			target_idx[idx_t] = k;
		}
	}
	
	if(all(tile_idx == n_tiles - 1))
	{
		for(int k = end + local_id; k < np; k += stride)
		{
			int2 part_idx = part_cell_idx[k];
			int2 target_tile = part_idx / TILE_SIZE;
			int idx_s = atomic_inc(&counter[target_tile.x + target_tile.y * n_tiles.x]);

			temp_cell_idx[idx_s] = part_idx;
			temp_velocities[idx_s] = part_velocities[k];
			temp_positions[idx_s] = part_positions[k];
		}
	}
}

__kernel void spec_inject_particles_opencl(__global int2 *restrict temp_cell_idx, __global float2 *restrict temp_positions, 
						   __global float3 *restrict temp_velocities, __global const int2 *restrict new_cell_idx, 
						   __global const float2 *restrict new_positions, __global const float3 *restrict new_velocities,
						   __global int *restrict counter, const int np, const int np_inj, const int2 n_tiles)
{
	const int thread_id = get_global_id(0);
	const int stride = get_global_size(0);
	
	for(int k = thread_id; k < np_inj; k += stride)
	{
		int2 part_idx = new_cell_idx[k];
		int2 target_tile = part_idx / TILE_SIZE;
		int idx_s = atomic_inc(&counter[target_tile.x + target_tile.y * n_tiles.x]);
		
		temp_cell_idx[idx_s] = part_idx;
		temp_velocities[idx_s] = new_velocities[k];
		temp_positions[idx_s] = new_positions[k];
	}
}

__kernel void spec_sort_2(__global int2 *restrict part_cell_idx, __global float2 *restrict part_positions, 
						  __global float3 *restrict part_velocities, __global const int2 *restrict temp_cell_idx, 
						  __global const float2 *restrict temp_positions, __global const float3 *restrict temp_velocities,
						  __global const int *restrict target_idx, __global const int *counter, 
						  __global const int *restrict temp_offset, const int2 n_tiles, const int sort_size, 
						  const int np_max)
{
	const int local_id = get_local_id(0);
	const int group_id = get_group_id(0);
	const int stride = get_local_size(0);
	
	const int begin = temp_offset[group_id];
	const int end = counter[group_id];
	
	for(int i = begin + local_id; i < end; i += stride)
	{
		part_cell_idx[target_idx[i]] = temp_cell_idx[i];
		part_velocities[target_idx[i]] = temp_velocities[i];
		part_positions[target_idx[i]] = temp_positions[i];
	}
}

/*********************************************************************************************
 Particle advance
 *********************************************************************************************/
inline void interpolate_fld_opencl(__local const float3 E[(TILE_SIZE + 2)][(TILE_SIZE + 2)], 
								   __local const float3 B[(TILE_SIZE + 2)][(TILE_SIZE + 2)],
								   const int ix, const int iy, const float x, const float y, 
								   float3 *restrict const Ep, float3 *restrict const Bp)
{
	const int ih = ix + ((x < 0.5f) ? -1 : 0);
	const int jh = iy + ((y < 0.5f) ? -1 : 0);
	
	const float w1h = x + ((x < 0.5f) ? 0.5f : -0.5f);
	const float w2h = y + ((y < 0.5f) ? 0.5f : -0.5f);
	
	Ep->x = (E[iy][ih].x * (1.0f - w1h) + E[iy][ih + 1].x * w1h) * (1.0f - y)
	+ (E[iy + 1][ih].x * (1.0f - w1h) + E[iy + 1][ih + 1].x * w1h) * y;
	Ep->y = (E[jh][ix].y * (1.0f - x) + E[jh][ix + 1].y * x) * (1.0f - w2h)
	+ (E[jh + 1][ix].y * (1.0f - x) + E[jh + 1][ix + 1].y * x) * w2h;
	Ep->z = (E[iy][ix].z * (1.0f - x) + E[iy][ix + 1].z * x) * (1.0f - y)
	+ (E[iy + 1][ix].z * (1.0f - x) + E[iy + 1][ix + 1].z * x) * y;
	
	Bp->x = (B[jh][ix].x * (1.0f - x) + B[jh][ix + 1].x * x) * (1.0f - w2h)
	+ (B[jh + 1][ix].x * (1.0f - x) + B[jh + 1][ix + 1].x * x) * w2h;
	Bp->y = (B[iy][ih].y * (1.0f - w1h) + B[iy][ih + 1].y * w1h) * (1.0f - y)
	+ (B[iy + 1][ih].y * (1.0f - w1h) + B[iy + 1][ih + 1].y * w1h) * y;
	Bp->z = (B[jh][ih].z * (1.0f - w1h) + B[jh][ih + 1].z * w1h) * (1.0f - w2h)
	+ (B[jh + 1][ih].z * (1.0f - w1h) + B[jh + 1][ih + 1].z * w1h)
	* w2h;
}

// Current deposition (adapted Villasenor-Bunemann method)
inline void dep_current_opencl(int ix, int iy, int di, int dj, float x0, float y0, float dx,
							   float dy, float qnx, float qny, float qvz_p, 
							   __local float3 J[(TILE_SIZE + 3)][(TILE_SIZE + 3)])
{
	t_vp vp[3];
	float qvz[3];
	int vnp = 1;

	// split
	vp[0].x0 = x0;
	vp[0].y0 = y0;

	vp[0].dx = dx;
	vp[0].dy = dy;

	vp[0].x1 = x0 + dx;
	vp[0].y1 = y0 + dy;

	qvz[0] = qvz_p / 2.0;

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

		qvz[1] = qvz[0] * delta;

		// Correct previous particle
		vp[0].x1 = ib;
		vp[0].dx *= (1.0f - delta);

		vp[0].dy *= (1.0f - delta);
		vp[0].y1 = ycross;

		qvz[0] *= (1.0f - delta);

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

		qvz[vnp] = qvz[isy] * delta;

		// Correct previous particle
		vp[isy].y1 = jb;
		vp[isy].dy *= (1.0f - delta);

		vp[isy].dx *= (1.0f - delta);
		vp[isy].x1 = xcross;

		qvz[isy] *= (1.0f - delta);

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
		float3 value;
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

		value.x = wl1 * wp1[0];
		value.y = wl2 * wp2[0];
		value.z = qvz[k] * (S0x[0] * S0y[0] + S1x[0] * S1y[0] 
			+ (S0x[0] * S1y[0] - S1x[0] * S0y[0]) / 2.0f);
		atomic_add_float3_local((__local float3_union *) &J[vp[k].iy][vp[k].ix], value);

		value.x = 0;
		value.y = wl2 * wp2[1];
		value.z = qvz[k] * (S0x[1] * S0y[0] + S1x[1] * S1y[0]
			 + (S0x[1] * S1y[0] - S1x[1] * S0y[0]) / 2.0f);
		atomic_add_float3_local((__local float3_union *) &J[vp[k].iy][vp[k].ix + 1], value);
		
		value.x = wl1 * wp1[1];
		value.y = 0;
		value.z = qvz[k] * (S0x[0] * S0y[1] + S1x[0] * S1y[1] 
			+ (S0x[0] * S1y[1] - S1x[0] * S0y[1]) / 2.0f);
		atomic_add_float3_local((__local float3_union *) &J[vp[k].iy + 1][vp[k].ix], value);
		
		value.x = 0;
		value.y = 0;
		value.z = qvz[k] * (S0x[1] * S0y[1] + S1x[1] * S1y[1] 
			+ (S0x[1] * S1y[1] - S1x[1] * S0y[1]) / 2.0f);
		atomic_add_float3_local((__local float3_union *) &J[vp[k].iy + 1][vp[k].ix + 1], value);
	}
}

__kernel void spec_advance_opencl(__global int2 *restrict part_cell_idx, __global float2 *restrict part_positions, 
								  __global float3 *restrict part_velocities, __global const int *restrict tile_offset,
								  __global int *restrict np_per_tile, __global int *restrict leaving_part, const int np_max,
								  __global const float3 *restrict E_buf, __global const float3 *restrict B_buf, 
								  __global float3 *restrict J_buf, const int nrow, const int field_size, 
								  const float tem, const float dt_dx, const float dt_dy, const float qnx, 
								  const float qny, const float q, const int nx0, const int nx1, const int2 n_tiles,
								  const int moving_window, const int shift)
{
	const int local_id = get_local_id(0);
	const int stride = get_local_size(0);
	const int2 tile_idx = (int2) (get_group_id(0), get_group_id(1));
	
	const int current_tile = tile_idx.x + tile_idx.y * n_tiles.x;
	const int begin = tile_offset[current_tile];
	const int end = tile_offset[current_tile + 1];
	
	__local int local_count[9];
	
	__local float3 E[(TILE_SIZE + 2)][(TILE_SIZE + 2)];
	__local float3 B[(TILE_SIZE + 2)][(TILE_SIZE + 2)];
	__local float3 J[(TILE_SIZE + 3)][(TILE_SIZE + 3)];
	
	for(int k = local_id; k < 9; k += stride)
		local_count[k] = 0;
	
	for(int k = local_id; k < (TILE_SIZE + 2) * (TILE_SIZE + 2); k += stride)
	{
		int2 local_idx = (int2) (k % (TILE_SIZE + 2), k / (TILE_SIZE + 2));
		int2 global_idx = local_idx + tile_idx * TILE_SIZE;
		
		E[local_idx.y][local_idx.x] = E_buf[global_idx.x + global_idx.y * nrow];
		B[local_idx.y][local_idx.x] = B_buf[global_idx.x + global_idx.y * nrow];
	}
	
	for(int k = local_id; k < (TILE_SIZE + 3) * (TILE_SIZE + 3); k += stride)
	{
		int2 local_idx = (int2) (k % (TILE_SIZE + 3), k / (TILE_SIZE + 3));
		J[local_idx.y][local_idx.x].x = 0.0f;
		J[local_idx.y][local_idx.x].y = 0.0f;
		J[local_idx.y][local_idx.x].z = 0.0f;
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
		
	for(int k = begin + local_id; k < end; k += stride)
    {
		float3 part_velocity = part_velocities[k];
		float2 part_pos = part_positions[k];
		int2 global_part_idx = part_cell_idx[k];
		int2 local_part_idx = global_part_idx - (tile_idx * TILE_SIZE - 1);
		
		float3 Ep, Bp;

		interpolate_fld_opencl(E, B, local_part_idx.x, local_part_idx.y, part_pos.x, part_pos.y, &Ep, &Bp);
		
		Ep *= tem;
		
		float3 ut = part_velocity + Ep;
		
		float ustq = ut.x * ut.x + ut.y * ut.y + ut.z * ut.z;
		float gtem = tem / sqrt(1.0f + ustq);
		
		Bp *= gtem;
		
		part_velocity.x = ut.x + ut.y * Bp.z - ut.z * Bp.y;
		part_velocity.y = ut.y + ut.z * Bp.x - ut.x * Bp.z;
		part_velocity.z = ut.z + ut.x * Bp.y - ut.y * Bp.x;
		
		float Bp_mag = Bp.x * Bp.x + Bp.y * Bp.y + Bp.z * Bp.z;
		float otsq = 2.0f / (1.0f + Bp_mag);
		
		Bp *= otsq;
		
		ut.x += part_velocity.y * Bp.z - part_velocity.z * Bp.y;
		ut.y += part_velocity.z * Bp.x - part_velocity.x * Bp.z;
		ut.z += part_velocity.x * Bp.y - part_velocity.y * Bp.x;
		
		part_velocity = ut + Ep;

		float usq = part_velocity.x * part_velocity.x + part_velocity.y * part_velocity.y
		        + part_velocity.z * part_velocity.z;
		float rg = 1.0f / sqrt(1.0f + usq);

		float dx = dt_dx * rg * part_velocity.x;
		float dy = dt_dy * rg * part_velocity.y;

		float x1 = part_pos.x + dx;
		float y1 = part_pos.y + dy;

		int di = LTRIM(x1);
		int dj = LTRIM(y1);

		float qvz = q * part_velocity.z * rg;

		dep_current_opencl(local_part_idx.x, local_part_idx.y, di, dj, part_pos.x, part_pos.y, dx, dy, qnx,
						   qny, qvz, J);

		global_part_idx.x += di;
		global_part_idx.y += dj;
				
		if (!moving_window)
		{
			if(global_part_idx.x < 0) global_part_idx.x += nx0;
			else if(global_part_idx.x >= nx0) global_part_idx.x -= nx0;
		}else if(moving_window && shift) global_part_idx.x--;
		
		if(global_part_idx.y < 0) global_part_idx.y += nx1;
		else if(global_part_idx.y >= nx1) global_part_idx.y -= nx1;
			
		part_positions[k].x = x1 - di;
		part_positions[k].y = y1 - dj;
		part_cell_idx[k] = global_part_idx;
		part_velocities[k] = part_velocity;		
		
		if(global_part_idx.x >= 0 && global_part_idx.x < nx0)
		{	
			int2 tile = global_part_idx / TILE_SIZE;
			int2 local_idx;
			
			if (tile_idx.x == n_tiles.x - 1 && tile.x == 0) local_idx.x = 2;
			else if (tile_idx.x == 0 && tile.x == n_tiles.x - 1) local_idx.x = 0;
			else local_idx.x = tile.x - tile_idx.x + 1;
			
			if (tile_idx.y == n_tiles.y - 1 && tile.y == 0) local_idx.y = 2;
			else if (tile_idx.y == 0 && tile.y == n_tiles.y - 1) local_idx.y = 0;
			else local_idx.y = tile.y - tile_idx.y + 1;
			
			atomic_inc(&(local_count[local_idx.x + local_idx.y * 3]));
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(int k = local_id; k < 9; k += stride)
	{
		int2 idx = (int2)(k / 3, k % 3);
		int2 global_idx = tile_idx + idx - 1; 
		
		if (global_idx.x < 0) global_idx.x += n_tiles.x;
		else if (global_idx.x >= n_tiles.x) global_idx.x -= n_tiles.x;
		
		if (global_idx.y < 0) global_idx.y += n_tiles.y;
		else if (global_idx.y >= n_tiles.y) global_idx.y -= n_tiles.y;
		
		if(local_count[idx.x + idx.y * 3] > 0)
			atomic_add(&np_per_tile[global_idx.x + global_idx.y * n_tiles.x], local_count[idx.x + idx.y * 3]);
	}
	
	for(int k = local_id; k < (TILE_SIZE + 3) * (TILE_SIZE + 3); k += stride)
	{
		int2 local_idx = (int2) (k % (TILE_SIZE + 3), k / (TILE_SIZE + 3));
		int2 global_idx = local_idx + tile_idx * TILE_SIZE;
		
		atomic_add_float3_global(&J_buf[global_idx.x + global_idx.y * nrow], J[local_idx.y][local_idx.x]);
	}
	
	if(local_id == 0) leaving_part[current_tile] = end - begin - local_count[4]; 
}
