/*********************************************************************************************
 ZPIC
 kernel_particles.c

 Created by Nicolas Guidotti on 14/06/2020

 Copyright 2020 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "particles.h"
#include "math.h"
#include "utilities.h"

#define MIN_WARP_SIZE 32

typedef struct {
	float x0, x1, y0, y1, dx, dy, qvz;
	int ix, iy;
} t_vp;

/*********************************************************************************************
 Utilities
 *********************************************************************************************/

// Prefetching for particle vector
#ifdef ENABLE_PREFETCH
void spec_prefetch_openacc(t_particle_vector *part, const int device)
{
	cudaMemPrefetchAsync(part->ix, part->size_max * sizeof(int), device, NULL);
	cudaMemPrefetchAsync(part->iy, part->size_max * sizeof(int), device, NULL);
	cudaMemPrefetchAsync(part->x, part->size_max * sizeof(t_part_data), device, NULL);
	cudaMemPrefetchAsync(part->y, part->size_max * sizeof(t_part_data), device, NULL);
	cudaMemPrefetchAsync(part->ux, part->size_max * sizeof(t_part_data), device, NULL);
	cudaMemPrefetchAsync(part->uy, part->size_max * sizeof(t_part_data), device, NULL);
	cudaMemPrefetchAsync(part->uz, part->size_max * sizeof(t_part_data), device, NULL);
	cudaMemPrefetchAsync(part->invalid, part->size_max * sizeof(bool), device, NULL);
}
#endif


// Add the block offset to the vector
#pragma oss task device(openacc) inout(vector[0; vector_size]) inout(block_sum[0; num_blocks]) \
	label("Prefix Sum (GPU, Add)")
void add_block_sum(int *restrict vector, int *restrict block_sum, const int num_blocks,
        const int block_size, const int vector_size, const int device)
{

#ifdef MANUAL_GPU_SETUP
	acc_set_device_num(device, DEVICE_TYPE);
#endif

	#pragma acc parallel loop gang
	for (int block_id = 1; block_id < num_blocks; block_id++)
	{
		const int begin_idx = block_id * block_size;

		#pragma acc loop vector
		for (int i = 0; i < block_size; i++)
			if (i + begin_idx < vector_size) vector[i + begin_idx] += block_sum[block_id];
	}
}

// Prefix Sum (Exclusive) - 1 warp per thread block
#pragma oss task device(openacc) inout(vector[0; size]) out(block_sum[0; num_blocks]) \
		label("Prefix Sum (GPU, Min Scan)")
void prefix_sum_min(int *restrict vector, int *restrict block_sum, const int num_blocks,
        const int size, const int device)
{
#ifdef MANUAL_GPU_SETUP
	acc_set_device_num(device, DEVICE_TYPE);
#endif

	// Prefix sum using a binomial tree
	#pragma acc parallel loop gang vector_length(MIN_WARP_SIZE)
	for (int block_id = 0; block_id < num_blocks; block_id++)
	{
		const int begin_idx = block_id * MIN_WARP_SIZE;
		int local_buffer[MIN_WARP_SIZE];

		#pragma acc cache(local_buffer[0: MIN_WARP_SIZE])

		// Copy to the local buffer
		#pragma acc loop vector
		for (int i = 0; i < MIN_WARP_SIZE; i++)
		{
			if (i + begin_idx < size) local_buffer[i] = vector[i + begin_idx];
			else local_buffer[i] = 0;
		}

		// Scan the tree upward (in direction to the root).
		// Add the values of each node and stores the result in the right node
		for (int offset = 1; offset < MIN_WARP_SIZE; offset *= 2)
		{
			#pragma acc loop vector
			for (int i = offset - 1; i < MIN_WARP_SIZE; i += 2 * offset)
				local_buffer[i + offset] += local_buffer[i];
		}

		// Store the total sum in the block sum vector and reset the last position of the vector
		block_sum[block_id] = local_buffer[MIN_WARP_SIZE - 1];
		local_buffer[MIN_WARP_SIZE - 1] = 0;

		// Scan the tree downward (from the root).
		// First, swap the values between nodes, then update the right node with the sum
		for (int offset = MIN_WARP_SIZE >> 1; offset > 0; offset >>= 1)
		{
			#pragma acc loop vector
			for (int i = offset - 1; i < MIN_WARP_SIZE; i += 2 * offset)
			{
				int temp = local_buffer[i];
				local_buffer[i] = local_buffer[i + offset];
				local_buffer[i + offset] += temp;
			}
		}

		// Store the results in the global vector
		#pragma acc loop vector
		for (int i = 0; i < MIN_WARP_SIZE; i++)
			if (i + begin_idx < size) vector[i + begin_idx] = local_buffer[i];
	}
}

// Prefix Sum (Exclusive) - Multiple warps per thread block
#pragma oss task device(openacc) inout(vector[0; size]) out(block_sum[0; num_blocks]) \
		label("Prefix Sum (GPU, Full)")
void prefix_sum_full(int *restrict vector, int *restrict block_sum, const int num_blocks,
        const int size, const int device)
{
#ifdef MANUAL_GPU_SETUP
	acc_set_device_num(device, DEVICE_TYPE);
#endif

	// Prefix sum using a binomial tree
	#pragma acc parallel loop gang vector_length(LOCAL_BUFFER_SIZE / 2)
	for (int block_id = 0; block_id < num_blocks; block_id++)
	{
		const int begin_idx = block_id * LOCAL_BUFFER_SIZE;
		int local_buffer[LOCAL_BUFFER_SIZE];

		#pragma acc cache(local_buffer[0: LOCAL_BUFFER_SIZE])

		// Copy to the local buffer
		#pragma acc loop vector
		for (int i = 0; i < LOCAL_BUFFER_SIZE; i++)
		{
			if (i + begin_idx < size) local_buffer[i] = vector[i + begin_idx];
			else local_buffer[i] = 0;
		}

		// Scan the tree upward (in direction to the root).
		// Add the values of each node and stores the result in the right node
		for (int offset = 1; offset < LOCAL_BUFFER_SIZE; offset *= 2)
		{
			#pragma acc loop vector
			for (int i = offset - 1; i < LOCAL_BUFFER_SIZE; i += 2 * offset)
				local_buffer[i + offset] += local_buffer[i];

		}

		// Store the total sum in the block sum vector and reset the last position of the vector
		block_sum[block_id] = local_buffer[LOCAL_BUFFER_SIZE - 1];
		local_buffer[LOCAL_BUFFER_SIZE - 1] = 0;

		// Scan the tree downward (from the root).
		// First, swap the values between nodes, then update the right node with the sum
		for (int offset = LOCAL_BUFFER_SIZE >> 1; offset > 0; offset >>= 1)
		{
			#pragma acc loop vector
			for (int i = offset - 1; i < LOCAL_BUFFER_SIZE; i += 2 * offset)
			{
				int temp = local_buffer[i];
				local_buffer[i] = local_buffer[i + offset];
				local_buffer[i + offset] += temp;
			}
		}

		// Store the results in the global vector
		#pragma acc loop vector
		for (int i = 0; i < LOCAL_BUFFER_SIZE; i++)
			if (i + begin_idx < size) vector[i + begin_idx] = local_buffer[i];
	}
}

// Prefix/Scan Sum (Exclusive)
void prefix_sum_openacc(int *restrict vector, const int size, const int device)
{
	int num_blocks;
	int *restrict block_sum;

	if(size < LOCAL_BUFFER_SIZE / 4)
	{
		num_blocks = ceil((float) size / MIN_WARP_SIZE);
		block_sum = malloc(num_blocks * sizeof(int));

		prefix_sum_min(vector, block_sum, num_blocks, size, device);

		if (num_blocks > 1)
		{
			prefix_sum_openacc(block_sum, num_blocks, device);

			// Add the values from the block sum
			add_block_sum(vector, block_sum, num_blocks, MIN_WARP_SIZE, size, device);
		}
	} else
	{
		num_blocks = ceil((float) size / LOCAL_BUFFER_SIZE);
		block_sum = malloc(num_blocks * sizeof(int));

		prefix_sum_full(vector, block_sum, num_blocks, size, device);

		if (num_blocks > 1)
		{
			prefix_sum_openacc(block_sum, num_blocks, device);

			// Add the values from the block sum
			add_block_sum(vector, block_sum, num_blocks, LOCAL_BUFFER_SIZE, size, device);
		}
	}

	#pragma oss taskwait on(block_sum[0; num_blocks])
	free(block_sum);
}

// Sort vector based on the new_pos array
void spec_move_vector_int_full(int *restrict vector, int *restrict new_pos, const int size)
{
	int *restrict temp = malloc(size * sizeof(int));

	#pragma acc parallel loop
	for(int i = 0; i < size; i++) temp[i] = vector[i];

	#pragma acc parallel loop
	for(int i = 0; i < size; i++)
		if(new_pos[i] >= 0) vector[new_pos[i]] = temp[i];

	free(temp);
}

// Apply the sorting to one of the particle vectors. If source_idx == NULL, apply the sorting in the whole array
#pragma oss task device(openacc) inout(*vector) inout(temp[0; move_size]) inout(source_idx[0; move_size]) \
	inout(target_idx[0; move_size]) label("Sort (GPU, Sort INT)")
void spec_move_vector_int(int *restrict vector, int *restrict temp, int *restrict source_idx,
        int *restrict target_idx, const int move_size, const int device)
{
#ifdef MANUAL_GPU_SETUP
	acc_set_device_num(device, DEVICE_TYPE);
#endif

	#pragma acc parallel loop
	for (int i = 0; i < move_size; i++)
		if (source_idx[i] >= 0) temp[i] = vector[source_idx[i]];

	#pragma acc parallel loop
	for (int i = 0; i < move_size; i++)
		if (source_idx[i] >= 0) vector[target_idx[i]] = temp[i];


}

// Sort vector based on the new_pos array
void spec_move_vector_float_full(float *restrict vector, int *restrict new_pos, const int size)
{
	float *restrict temp = malloc(size * sizeof(float));

	#pragma acc parallel loop
	for(int i = 0; i < size; i++) temp[i] = vector[i];

	#pragma acc parallel loop
	for(int i = 0; i < size; i++)
		if(new_pos[i] >= 0) vector[new_pos[i]] = temp[i];

	free(temp);
}

// Apply the sorting to one of the particle vectors. If source_idx == NULL, apply the sorting in the whole array
#pragma oss task device(openacc) inout(*vector) inout(temp[0; move_size]) inout(source_idx[0; move_size]) \
	inout(target_idx[0; move_size]) label("Sort (GPU, Sort FLOAT)")
void spec_move_vector_float(float *restrict vector, float *restrict temp, int *restrict source_idx,
        int *restrict target_idx, const int move_size, const int device)
{

#ifdef MANUAL_GPU_SETUP
	acc_set_device_num(device, DEVICE_TYPE);
#endif

	#pragma acc parallel loop
	for (int i = 0; i < move_size; i++)
		if (source_idx[i] >= 0) temp[i] = vector[source_idx[i]];

	#pragma acc parallel loop
	for (int i = 0; i < move_size; i++)
		if (source_idx[i] >= 0) vector[target_idx[i]] = temp[i];


}

/*********************************************************************************************
 Initialisation
 *********************************************************************************************/

// Organize the particles in tiles (Bucket Sort)
void spec_organize_in_tiles(t_species *spec, const int limits_y[2], const int device)
{
	int iy, ix;

	const int size = spec->main_vector.size;
	const int n_tiles_x = spec->n_tiles_x;
	const int n_tiles_y = spec->n_tiles_y;

	spec->mv_part_offset = calloc((n_tiles_y * n_tiles_x + 1), sizeof(int));
	spec->tile_offset = calloc((n_tiles_y * n_tiles_x + 1), sizeof(int));
	int *restrict tile_offset = spec->tile_offset;

	int *restrict pos = alloc_align_buffer(DEFAULT_ALIGNMENT, size * sizeof(int));

#ifdef MANUAL_GPU_SETUP
	acc_set_device_num(device, DEVICE_TYPE);
#endif

#ifdef ENABLE_PREFETCH
	spec_prefetch_openacc(&spec->main_vector, device);
	cudaMemPrefetchAsync(pos, size * sizeof(int), device, NULL);
	cudaMemPrefetchAsync(spec->tile_offset, (n_tiles_x * n_tiles_y + 1) * sizeof(int), device, NULL);
#endif

	// Calculate the histogram (number of particles per tile)
	#pragma acc parallel loop private(ix, iy)
	for (int i = 0; i < size; i++)
	{
		ix = spec->main_vector.ix[i] / TILE_SIZE;
		iy = (spec->main_vector.iy[i] - limits_y[0]) / TILE_SIZE;

		#pragma acc atomic capture
		pos[i] = tile_offset[ix + iy * n_tiles_x]++;
	}

	// Prefix sum to find the initial idx of each tile in the particle vector
	prefix_sum_openacc(tile_offset, n_tiles_x * n_tiles_y + 1, device);

	// Calculate the target position of each particle
	#pragma acc parallel loop private(ix, iy)
	for (int i = 0; i < size; i++)
	{
		ix = spec->main_vector.ix[i] / TILE_SIZE;
		iy = (spec->main_vector.iy[i] - limits_y[0]) / TILE_SIZE;

		pos[i] += tile_offset[ix + iy * n_tiles_x];
	}

	const int final_size = tile_offset[n_tiles_x * n_tiles_y];
	spec->main_vector.size = final_size;

	// Move the particles to the correct position
	spec_move_vector_int_full(spec->main_vector.ix, pos, size);
	spec_move_vector_int_full(spec->main_vector.iy, pos, size);
	spec_move_vector_float_full(spec->main_vector.x, pos, size);
	spec_move_vector_float_full(spec->main_vector.y, pos, size);
	spec_move_vector_float_full(spec->main_vector.ux, pos, size);
	spec_move_vector_float_full(spec->main_vector.uy, pos, size);
	spec_move_vector_float_full(spec->main_vector.uz, pos, size);

	// Validate all the particles
	#pragma acc parallel loop
	for (int k = 0; k < final_size; k++)
		spec->main_vector.invalid[k] = false;

	free_align_buffer(pos);  // Clean position vector
}

/*********************************************************************************************
 Particle Advance
 *********************************************************************************************/

// EM fields interpolation. OpenAcc
//#pragma acc routine
void interpolate_fld_openacc(const t_vfld *restrict const E, const t_vfld *restrict const B,
		const int nrow, const int ix, const int iy, const t_fld x, const t_fld y,
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
			+ (B[ih + (jh + 1) * nrow].z * (1.0f - w1h) + B[ih + 1 + (jh + 1) * nrow].z * w1h)
					* w2h;
}

// Current deposition (adapted Villasenor-Bunemann method). OpenAcc
//#pragma acc routine
void dep_current_openacc(int ix, int iy, int di, int dj, float x0, float y0, float dx,
		float dy, float qnx, float qny, float qvz, t_vfld *restrict const J, const int nrow,
		t_vp vp[THREAD_BLOCK * 3], const int thread_id)
{
	const int begin = thread_id * 3;

	// Split the particle trajectory
	int vnp = 1;

	// split
	vp[begin].x0 = x0;
	vp[begin].y0 = y0;

	vp[begin].dx = dx;
	vp[begin].dy = dy;

	vp[begin].x1 = x0 + dx;
	vp[begin].y1 = y0 + dy;

	vp[begin].qvz = qvz / 2.0;

	vp[begin].ix = ix;
	vp[begin].iy = iy;

	// x split
	if (di != 0)
	{
		//int ib = ( di+1 )>>1;
		int ib = (di == 1);

		float delta = (x0 + dx - ib) / dx;

		// Add new particle
		vp[begin + 1].x0 = 1 - ib;
		vp[begin + 1].x1 = (x0 + dx) - di;
		vp[begin + 1].dx = dx * delta;
		vp[begin + 1].ix = ix + di;

		float ycross = y0 + dy * (1.0f - delta);

		vp[begin + 1].y0 = ycross;
		vp[begin + 1].y1 = vp[begin].y1;
		vp[begin + 1].dy = dy * delta;
		vp[begin + 1].iy = iy;

		vp[begin + 1].qvz = vp[begin].qvz * delta;

		// Correct previous particle
		vp[begin].x1 = ib;
		vp[begin].dx *= (1.0f - delta);

		vp[begin].dy *= (1.0f - delta);
		vp[begin].y1 = ycross;

		vp[begin].qvz *= (1.0f - delta);

		vnp++;
	}

	// ysplit
	if (dj != 0)
	{
		int isy = 1 - (vp[begin].y1 < 0.0f || vp[begin].y1 >= 1.0f);

		// int jb = ( dj+1 )>>1;
		int jb = (dj == 1);

		// The static analyser gets confused by this but it is correct
		float delta = (vp[begin + isy].y1 - jb) / vp[begin + isy].dy;

		// Add new particle
		vp[begin + vnp].y0 = 1 - jb;
		vp[begin + vnp].y1 = vp[begin + isy].y1 - dj;
		vp[begin + vnp].dy = vp[begin + isy].dy * delta;
		vp[begin + vnp].iy = vp[begin + isy].iy + dj;

		float xcross = vp[begin + isy].x0 + vp[begin + isy].dx * (1.0f - delta);

		vp[begin + vnp].x0 = xcross;
		vp[begin + vnp].x1 = vp[begin + isy].x1;
		vp[begin + vnp].dx = vp[begin + isy].dx * delta;
		vp[begin + vnp].ix = vp[begin + isy].ix;

		vp[begin + vnp].qvz = vp[begin + isy].qvz * delta;

		// Correct previous particle
		vp[begin + isy].y1 = jb;
		vp[begin + isy].dy *= (1.0f - delta);

		vp[begin + isy].dx *= (1.0f - delta);
		vp[begin + isy].x1 = xcross;

		vp[begin + isy].qvz *= (1.0f - delta);

		// Correct extra vp if needed
		if (isy < vnp - 1)
		{
			vp[begin + 1].y0 -= dj;
			vp[begin + 1].y1 -= dj;
			vp[begin + 1].iy += dj;
		}
		vnp++;
	}

	// Deposit virtual particle currents
	for (int k = begin; k < begin + vnp; k++)
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

		#pragma acc atomic
		J[vp[k].ix + nrow * vp[k].iy].x += wl1 * wp1[0];

		#pragma acc atomic
		J[vp[k].ix + nrow * (vp[k].iy + 1)].x += wl1 * wp1[1];

		#pragma acc atomic
		J[vp[k].ix + nrow * vp[k].iy].y += wl2 * wp2[0];

		#pragma acc atomic
		J[vp[k].ix + 1 + nrow * vp[k].iy].y += wl2 * wp2[1];

		#pragma acc atomic
		J[vp[k].ix + nrow * vp[k].iy].z += vp[k].qvz
				* (S0x[0] * S0y[0] + S1x[0] * S1y[0] + (S0x[0] * S1y[0] - S1x[0] * S0y[0]) / 2.0f);

		#pragma acc atomic
		J[vp[k].ix + 1 + nrow * vp[k].iy].z += vp[k].qvz
				* (S0x[1] * S0y[0] + S1x[1] * S1y[0] + (S0x[1] * S1y[0] - S1x[1] * S0y[0]) / 2.0f);

		#pragma acc atomic
		J[vp[k].ix + nrow * (vp[k].iy + 1)].z += vp[k].qvz
				* (S0x[0] * S0y[1] + S1x[0] * S1y[1] + (S0x[0] * S1y[1] - S1x[0] * S0y[1]) / 2.0f);

		#pragma acc atomic
		J[vp[k].ix + 1 + nrow * (vp[k].iy + 1)].z += vp[k].qvz
				* (S0x[1] * S0y[1] + S1x[1] * S1y[1] + (S0x[1] * S1y[1] - S1x[1] * S0y[1]) / 2.0f);
	}
}

// Advance u using Boris scheme. OpenAcc
//#pragma acc routine
void advance_part_momentum(t_float3 *part_velocity, t_vfld Ep, t_vfld Bp, const t_part_data tem)
{
	Ep.x *= tem;
	Ep.y *= tem;
	Ep.z *= tem;

	t_float3 ut;
	ut.x = part_velocity->x + Ep.x;
	ut.y = part_velocity->y + Ep.y;
	ut.z = part_velocity->z + Ep.z;

	// Perform first half of the rotation
	t_part_data ustq = ut.x * ut.x + ut.y * ut.y + ut.z * ut.z;
	t_part_data gtem = tem / sqrtf(1.0f + ustq);

	Bp.x *= gtem;
	Bp.y *= gtem;
	Bp.z *= gtem;

	part_velocity->x = ut.x + ut.y * Bp.z - ut.z * Bp.y;
	part_velocity->y = ut.y + ut.z * Bp.x - ut.x * Bp.z;
	part_velocity->z = ut.z + ut.x * Bp.y - ut.y * Bp.x;

	// Perform second half of the rotation
	t_part_data Bp_mag = Bp.x * Bp.x + Bp.y * Bp.y + Bp.z * Bp.z;
	t_part_data otsq = 2.0f / (1.0f + Bp_mag);

	Bp.x *= otsq;
	Bp.y *= otsq;
	Bp.z *= otsq;

	ut.x += part_velocity->y * Bp.z - part_velocity->z * Bp.y;
	ut.y += part_velocity->z * Bp.x - part_velocity->x * Bp.z;
	ut.z += part_velocity->x * Bp.y - part_velocity->y * Bp.x;

	// Perform second half of electric field acceleration
	part_velocity->x = ut.x + Ep.x;
	part_velocity->y = ut.y + Ep.y;
	part_velocity->z = ut.z + Ep.z;
}

// Particle advance (OpenAcc). Optimised for GPU architecture
void spec_advance_openacc(t_species *restrict const spec, const t_emf *restrict const emf,
		t_current *restrict const current, const int limits_y[2], const int device)
{
	const t_part_data tem = 0.5 * spec->dt / spec->m_q;
	const t_part_data dt_dx = spec->dt / spec->dx[0];
	const t_part_data dt_dy = spec->dt / spec->dx[1];

	// Auxiliary values for current deposition
	const t_part_data qnx = spec->q * spec->dx[0] / spec->dt;
	const t_part_data qny = spec->q * spec->dx[1] / spec->dt;

	const int nrow = emf->nrow;
	const int region_offset = limits_y[0];

#ifdef MANUAL_GPU_SETUP
	acc_set_device_num(device, DEVICE_TYPE);
#endif

#ifdef ENABLE_PREFETCH
	spec_prefetch_openacc(&spec->main_vector, device);
	cudaMemPrefetchAsync(spec->tile_offset, (spec->n_tiles_x * spec->n_tiles_y + 1) * sizeof(int), device, NULL);
	current_prefetch_openacc(current->J_buf, current->total_size, device);
	emf_prefetch_openacc(emf->B_buf, emf->total_size, device);
	emf_prefetch_openacc(emf->E_buf, emf->total_size, device);
#endif

	// Advance particles
	#pragma acc parallel loop gang collapse(2) vector_length(THREAD_BLOCK)
	for(int tile_y = 0; tile_y < spec->n_tiles_y; tile_y++)
	{
		for(int tile_x = 0; tile_x < spec->n_tiles_x; tile_x++)
		{
			const int tile_idx = tile_x + tile_y * spec->n_tiles_x;
			const int begin = spec->tile_offset[tile_idx];
			const int end = spec->tile_offset[tile_idx + 1];

			t_vfld E[(TILE_SIZE + 2) * (TILE_SIZE + 2)];
			t_vfld B[(TILE_SIZE + 2) * (TILE_SIZE + 2)];
			t_vfld J[(TILE_SIZE + 3) * (TILE_SIZE + 3)];

			#pragma acc cache(E[0 : (TILE_SIZE + 2) * (TILE_SIZE + 2)])
			#pragma acc cache(B[0 : (TILE_SIZE + 2) * (TILE_SIZE + 2)])
			#pragma acc cache(J[0 : (TILE_SIZE + 3) * (TILE_SIZE + 3)])

			t_vp vp[THREAD_BLOCK * 3];
			#pragma acc cache(vp[0 : THREAD_BLOCK * 3])

			// Load the EMF into the shared memory
			#pragma acc loop vector collapse(2)
			for(int j = 0; j < (TILE_SIZE + 2); j++)
			{
				for(int i = 0; i < (TILE_SIZE + 2); i++)
				{
					t_integer2 idx;
					idx.x = (tile_x * TILE_SIZE + i - 1);
					idx.y = (tile_y * TILE_SIZE + j - 1);

					E[i + j * (TILE_SIZE + 2)] = emf->E[idx.x + idx.y * nrow];
					B[i + j * (TILE_SIZE + 2)] = emf->B[idx.x + idx.y * nrow];
				}
			}

			// Reset the local current to 0
			#pragma acc loop vector
			for(int i = 0; i < (TILE_SIZE + 3) * (TILE_SIZE + 3); i++)
			{
				J[i].x = 0.0f;
				J[i].y = 0.0f;
				J[i].z = 0.0f;
			}

			#pragma acc loop vector
			for (int k = begin; k < end; k++)
			{
				// Load particle momenta into local variable
				t_float3 part_velocity;
				part_velocity.x = spec->main_vector.ux[k];
				part_velocity.y = spec->main_vector.uy[k];
				part_velocity.z = spec->main_vector.uz[k];

				// Load particle position into local variable
				t_float2 part_pos;
				part_pos.x = spec->main_vector.x[k];
				part_pos.y = spec->main_vector.y[k];

				// Load particle cell index into local variable
				// Then convert to local coordinates
				t_integer2 part_idx;
				part_idx.x = spec->main_vector.ix[k] - (tile_x * TILE_SIZE - 1);
				part_idx.y = spec->main_vector.iy[k] - (tile_y * TILE_SIZE - 1) - region_offset;

				t_vfld Ep, Bp;

				// Interpolate fields
				interpolate_fld_openacc(E, B, (TILE_SIZE + 2), part_idx.x, part_idx.y, part_pos.x,
				        part_pos.y, &Ep, &Bp);

				// Advance the particle momentum
				advance_part_momentum(&part_velocity, Ep, Bp, tem);

				// Push particle
				t_part_data usq = part_velocity.x * part_velocity.x
				        + part_velocity.y * part_velocity.y + part_velocity.z * part_velocity.z;
				t_part_data rg = 1.0f / sqrtf(1.0f + usq);

				t_part_data dx = dt_dx * rg * part_velocity.x;
				t_part_data dy = dt_dy * rg * part_velocity.y;

				t_part_data x1 = part_pos.x + dx;
				t_part_data y1 = part_pos.y + dy;

				int di = LTRIM(x1);
				int dj = LTRIM(y1);

				t_part_data qvz = spec->q * part_velocity.z * rg;

				// Deposit current
				dep_current_openacc(part_idx.x, part_idx.y, di, dj, part_pos.x, part_pos.y, dx, dy,
				        qnx, qny, qvz, J, (TILE_SIZE + 3), vp, k % THREAD_BLOCK);

				// Store results
				spec->main_vector.x[k] = x1 - di;
				spec->main_vector.y[k] = y1 - dj;
				spec->main_vector.ix[k] += di;
				spec->main_vector.iy[k] += dj;
				spec->main_vector.ux[k] = part_velocity.x;
				spec->main_vector.uy[k] = part_velocity.y;
				spec->main_vector.uz[k] = part_velocity.z;
			}

			// Update the global current with local values
			#pragma acc loop vector collapse(2)
			for(int j = 0; j < (TILE_SIZE + 3); j++)
			{
				for(int i = 0; i < (TILE_SIZE + 3); i++)
				{
					t_integer2 idx;
					idx.x = (tile_x * TILE_SIZE + i - 1);
					idx.y = (tile_y * TILE_SIZE + j - 1);

					#pragma acc atomic
					current->J[idx.x + idx.y * nrow].x += J[i + j * ((TILE_SIZE + 3))].x;

					#pragma acc atomic
					current->J[idx.x + idx.y * nrow].y += J[i + j * ((TILE_SIZE + 3))].y;

					#pragma acc atomic
					current->J[idx.x + idx.y * nrow].z += J[i + j * ((TILE_SIZE + 3))].z;
				}
			}
		}
	}

	// Advance internal iteration number
	spec->iter++;
}

// Particle advance (OpenAcc). Default implementation
void spec_advance_openacc_default(t_species *restrict const spec, const t_emf *restrict const emf,
		t_current *restrict const current, const int limits_y[2])
{
	const t_part_data tem = 0.5 * spec->dt / spec->m_q;
	const t_part_data dt_dx = spec->dt / spec->dx[0];
	const t_part_data dt_dy = spec->dt / spec->dx[1];

	// Auxiliary values for current deposition
	const t_part_data qnx = spec->q * spec->dx[0] / spec->dt;
	const t_part_data qny = spec->q * spec->dx[1] / spec->dt;

	const t_vfld *restrict E = emf->E;
	const t_vfld *restrict B = emf->B;
	t_vfld *restrict J = current->J;

	const int nrow = emf->nrow;
	const int region_offset = limits_y[0];

	// Advance particles
	#pragma acc parallel loop
	for (int k = 0; k < spec->main_vector.size; k++)
	{
		bool is_invalid = spec->main_vector.invalid[k];

		if (!is_invalid)
		{
			t_float3 part_velocity;
			part_velocity.x = spec->main_vector.ux[k];
			part_velocity.y = spec->main_vector.uy[k];
			part_velocity.z = spec->main_vector.uz[k];

			t_float2 part_pos;
			part_pos.x = spec->main_vector.x[k];
			part_pos.y = spec->main_vector.y[k];

			t_integer2 part_idx;
			part_idx.x = spec->main_vector.ix[k];
			part_idx.y = spec->main_vector.iy[k] - region_offset;

			t_vfld Ep, Bp;

			// Interpolate fields
			interpolate_fld_openacc(E, B, nrow, part_idx.x, part_idx.y, part_pos.x,
									part_pos.y, &Ep, &Bp);

			// Advance the particle momenta
			advance_part_momentum(&part_velocity, Ep, Bp, tem);

			// Push particle
			t_part_data usq = part_velocity.x * part_velocity.x + part_velocity.y * part_velocity.y
					+ part_velocity.z * part_velocity.z;
			t_part_data rg = 1.0f / sqrtf(1.0f + usq);

			t_part_data dx = dt_dx * rg * part_velocity.x;
			t_part_data dy = dt_dy * rg * part_velocity.y;

			t_part_data x1 = part_pos.x + dx;
			t_part_data y1 = part_pos.y + dy;

			int di = LTRIM(x1);
			int dj = LTRIM(y1);

			t_part_data qvz = spec->q * part_velocity.z * rg;

			t_vp vp[3];
			dep_current_openacc(part_idx.x, part_idx.y, di, dj, part_pos.x, part_pos.y, dx, dy, qnx,
								qny, qvz, J, nrow, vp, 0);

			// Store results
			spec->main_vector.x[k] = x1 - di;
			spec->main_vector.y[k] = y1 - dj;
			spec->main_vector.ix[k] += di;
			spec->main_vector.iy[k] += dj;
			spec->main_vector.ux[k] = part_velocity.x;
			spec->main_vector.uy[k] = part_velocity.y;
			spec->main_vector.uz[k] = part_velocity.z;
		}
	}

	// Advance internal iteration number
	spec->iter++;
}

/*********************************************************************************************
 Post Processing
 *********************************************************************************************/

// Shift the particle left and inject particles in the rightmost cells. OpenAcc Task
void spec_move_window_openacc(t_species *restrict spec, const int limits_y[2], const int device)
{
	// Move window
	if (spec->iter * spec->dt > spec->dx[0] * (spec->n_move + 1))
	{
		const int size = spec->main_vector.size;

#ifdef MANUAL_GPU_SETUP
	acc_set_device_num(device, DEVICE_TYPE);
#endif

#ifdef ENABLE_PREFETCH
		spec_prefetch_openacc(&spec->main_vector, device);
#endif

		// Shift particles left
		#pragma acc parallel loop
		for(int i = 0; i < size; i++)
			if(!spec->main_vector.invalid[i]) spec->main_vector.ix[i]--;

		// Increase moving window counter
		spec->n_move++;

		const int range[][2] = {{spec->nx[0] - 1, spec->nx[0]}, {limits_y[0], limits_y[1]}};
		int np_inj = (range[0][1] - range[0][0]) * (range[1][1] - range[1][0]) * spec->ppc[0] * spec->ppc[1];

		// If needed, add the incoming particles to a temporary vector
		if(!spec->incoming_part[2].enable_vector)
		{
			part_vector_alloc(&spec->incoming_part[2], np_inj);
			spec_inject_particles(&spec->incoming_part[2], range, spec->ppc, &spec->density,
					spec->dx, spec->n_move, spec->ufl, spec->uth);
		}else spec->incoming_part[2].size = np_inj; // Reuse the temporary vector (THIS ONLY WORKS IF THE INJECTED PARTICLES HAVE NO MOMENTUM)
	}
}

// Transfer particles between regions (if applicable). OpenAcc Task
void spec_check_boundaries_openacc(t_species *spec, const int limits_y[2], const int device)
{
	const int nx0 = spec->nx[0];
	const int nx1 = spec->nx[1];

#ifdef MANUAL_GPU_SETUP
	acc_set_device_num(device, DEVICE_TYPE);
#endif

#ifdef ENABLE_PREFETCH
	spec_prefetch_openacc(&spec->main_vector, device);
	cudaMemPrefetchAsync(spec->tile_offset, (spec->n_tiles_x * spec->n_tiles_y + 1) * sizeof(int), device, NULL);
	spec_prefetch_openacc(spec->outgoing_part[1], device);
	spec_prefetch_openacc(spec->outgoing_part[0], device);
#endif

	// Check if particles are exiting the left boundary (periodic boundary)
	#pragma acc parallel loop gang vector_length(128)
	for(int tile_y = 0; tile_y < spec->n_tiles_y; tile_y++)
	{
		const int tile_idx = tile_y * spec->n_tiles_x;
		const int begin = spec->tile_offset[tile_idx];
		const int end = spec->tile_offset[tile_idx + 1];

		if(spec->moving_window)
		{
			#pragma acc loop vector
			for(int i = begin; i < end; i++)
				if (spec->main_vector.ix[i] < 0) spec->main_vector.invalid[i] = true;  // Mark the particle as invalid
		}else
		{
			#pragma acc loop vector
			for(int i = begin; i < end; i++)
				if (spec->main_vector.ix[i] < 0) spec->main_vector.ix[i] += nx0;
		}
	}

	// Check if particles are exiting the right boundary (periodic boundary)
	#pragma acc parallel loop gang vector_length(128)
	for(int tile_y = 0; tile_y < spec->n_tiles_y; tile_y++)
	{
		const int tile_idx = (tile_y + 1) * spec->n_tiles_x - 1;
		const int begin = spec->tile_offset[tile_idx];
		const int end = spec->tile_offset[tile_idx + 1];

		if(spec->moving_window)
		{
			#pragma acc loop vector
			for(int i = begin; i < end; i++)
				if (spec->main_vector.ix[i] >= nx0) spec->main_vector.invalid[i] = true;  // Mark the particle as invalid
		}else
		{
			#pragma acc loop vector
			for(int i = begin; i < end; i++)
				if (spec->main_vector.ix[i] >= nx0) spec->main_vector.ix[i] -= nx0;
		}
	}

	// Check if particles are exiting the lower boundary and needs to be transfer to another region
	#pragma acc parallel loop gang
	for (int tile_x = 0; tile_x < spec->n_tiles_x; tile_x++)
	{
		const int begin = spec->tile_offset[tile_x];
		const int end = spec->tile_offset[tile_x + 1];

		#pragma acc loop vector
		for (int i = begin; i < end; i++)
		{
			bool is_invalid = spec->main_vector.invalid[i];

			if (!is_invalid)
			{
				int iy = spec->main_vector.iy[i];
				int idx;

				// Check if the particle is leaving the box
				if (iy < limits_y[0])
				{
					if (iy < 0) iy += nx1;

					// Reserve a position in the vector
					#pragma acc atomic capture
					idx = spec->outgoing_part[0]->size++;

					spec->outgoing_part[0]->ix[idx] = spec->main_vector.ix[i];
					spec->outgoing_part[0]->iy[idx] = iy;
					spec->outgoing_part[0]->x[idx] = spec->main_vector.x[i];
					spec->outgoing_part[0]->y[idx] = spec->main_vector.y[i];
					spec->outgoing_part[0]->ux[idx] = spec->main_vector.ux[i];
					spec->outgoing_part[0]->uy[idx] = spec->main_vector.uy[i];
					spec->outgoing_part[0]->uz[idx] = spec->main_vector.uz[i];
					spec->outgoing_part[0]->invalid[idx] = false;

					spec->main_vector.invalid[i] = true;  // Mark the particle as invalid
				}
			}
		}
	}

	// Check if particles are exiting the upper boundary and needs to be transfer to another region
	#pragma acc parallel loop gang
	for (int tile_x = 0; tile_x < spec->n_tiles_x; tile_x++)
	{
		const int tile_idx = tile_x + (spec->n_tiles_y - 1) * spec->n_tiles_x;
		const int begin = spec->tile_offset[tile_idx];
		const int end = spec->tile_offset[tile_idx + 1];

		#pragma acc loop vector
		for (int i = begin; i < end; i++)
		{
			if (!spec->main_vector.invalid[i])
			{
				int iy = spec->main_vector.iy[i];
				int idx;

				// Check if the particle is leaving the box
				if (iy >= limits_y[1])
				{
					if (iy >= nx1) iy -= nx1;

					#pragma acc atomic capture
					idx = spec->outgoing_part[1]->size++;

					spec->outgoing_part[1]->ix[idx] = spec->main_vector.ix[i];
					spec->outgoing_part[1]->iy[idx] = iy;
					spec->outgoing_part[1]->x[idx] = spec->main_vector.x[i];
					spec->outgoing_part[1]->y[idx] = spec->main_vector.y[i];
					spec->outgoing_part[1]->ux[idx] = spec->main_vector.ux[i];
					spec->outgoing_part[1]->uy[idx] = spec->main_vector.uy[i];
					spec->outgoing_part[1]->uz[idx] = spec->main_vector.uz[i];
					spec->outgoing_part[1]->invalid[idx] = false;

					spec->main_vector.invalid[i] = true;  // Mark the particle as invalid
				}
			}
		}
	}
}

/*********************************************************************************************
 Sort
 *********************************************************************************************/

// Bucket sort (Full).
void spec_full_sort_openacc(t_species *spec, const int limits_y[2])
{
	int iy, ix;

	const int size = spec->main_vector.size;
	const int n_tiles_x = spec->n_tiles_x;
	const int n_tiles_y = spec->n_tiles_y;

	if(!spec->tile_offset) spec->tile_offset = malloc((n_tiles_y * n_tiles_x + 1) * sizeof(int));

	int *restrict tile_offset = spec->tile_offset;
	int *restrict pos = alloc_align_buffer(DEFAULT_ALIGNMENT, size * sizeof(int));

	#pragma acc parallel loop
	for(int i = 0; i <= n_tiles_x * n_tiles_y; i++)
		tile_offset[i] = 0;

	// Calculate the histogram (number of particles per tile)
	#pragma acc parallel loop private(ix, iy)
	for (int i = 0; i < size; i++)
	{
		if(!spec->main_vector.invalid[i])
		{
			ix = spec->main_vector.ix[i] / TILE_SIZE;
			iy = (spec->main_vector.iy[i] - limits_y[0]) / TILE_SIZE;

			#pragma acc atomic capture
			{
				pos[i] = tile_offset[ix + iy * n_tiles_x];
				tile_offset[ix + iy * n_tiles_x]++;
			}
		}else pos[i] = -1;
	}

	// Prefix sum to find the initial idx of each tile in the particle vector
	prefix_sum_openacc(tile_offset, n_tiles_x * n_tiles_y + 1, 0);

	// Calculate the target position of each particle
	#pragma acc parallel loop private(ix, iy)
	for (int i = 0; i < size; i++)
	{
		if (pos[i] >= 0)
		{
			ix = spec->main_vector.ix[i] / TILE_SIZE;
			iy = (spec->main_vector.iy[i] - limits_y[0]) / TILE_SIZE;

			pos[i] += tile_offset[ix + iy * n_tiles_x];
		}
	}

	const int final_size = tile_offset[n_tiles_x * n_tiles_y];
	spec->main_vector.size = final_size;

	// Move the particles to the correct position
	spec_move_vector_int_full(spec->main_vector.ix, pos, size);
	spec_move_vector_int_full(spec->main_vector.iy, pos, size);
	spec_move_vector_float_full(spec->main_vector.x, pos, size);
	spec_move_vector_float_full(spec->main_vector.y, pos, size);
	spec_move_vector_float_full(spec->main_vector.ux, pos, size);
	spec_move_vector_float_full(spec->main_vector.uy, pos, size);
	spec_move_vector_float_full(spec->main_vector.uz, pos, size);

	// Validate all the particles
	#pragma acc parallel loop
	for (int k = 0; k < final_size; k++)
		spec->main_vector.invalid[k] = false;

	free_align_buffer(pos); // Clean position vector
}

// Calculate an histogram for the number of particles per tile
#pragma oss task device(openacc) in(*part_vector) in(incoming_part[0:1]) \
	inout(tile_offset[0: n_tiles_x * n_tiles_y]) label("Sort (GPU, Histogram NP)")
void histogram_np_per_tile(t_particle_vector *part_vector, int *restrict tile_offset,
		t_particle_vector incoming_part[3], int *restrict np_per_tile, const int n_tiles_y, const int n_tiles_x,
		const int offset_region, const int device)
{
	const int n_tiles = n_tiles_x * n_tiles_y;

#ifdef MANUAL_GPU_SETUP
	acc_set_device_num(device, DEVICE_TYPE);
#endif

	// Reset the number of particles per tile
	#pragma acc parallel loop
	for (int i = 0; i < n_tiles; i++)
		np_per_tile[i] = 0;

	// Calculate the histogram (number of particles per tile) for the main vector
	#pragma acc parallel loop gang collapse(2)
	for(int tile_y = 0; tile_y < n_tiles_y; tile_y++)
	{
		for(int tile_x = 0; tile_x < n_tiles_x; tile_x++)
		{
			const int tile_idx = tile_x + tile_y * n_tiles_x;
			const int begin = tile_offset[tile_idx];
			const int end = tile_offset[tile_idx + 1];

			// Use shared memory to calculate a local histogram
			int np[9];
			#pragma acc cache(np[0: 9])

			#pragma acc loop vector
			for(int i = 0; i < 9; i++)
				np[i] = 0;

			#pragma acc loop vector
			for (int k = begin; k < end; k++)
			{
				int ix = part_vector->ix[k] / TILE_SIZE;
				int iy = (part_vector->iy[k] - offset_region) / TILE_SIZE;
				bool is_invalid = part_vector->invalid[k];

				int local_ix;
				int local_iy = (iy - tile_y + 1);

				if (tile_x == n_tiles_x - 1 && ix == 0) local_ix = 2;
				else if (tile_x == 0 && ix == n_tiles_x - 1) local_ix = 0;
				else local_ix = (ix - tile_x + 1);

				if (!is_invalid)
				{
					#pragma acc atomic
					np[local_ix + local_iy * 3]++;
				}
			}

			// Add the local values to the a global histogram
			#pragma acc loop vector collapse(2)
			for(int j = 0; j < 3; j++)
			{
				for(int i = 0; i < 3; i++)
				{
					int global_ix = tile_x + i - 1;
					int global_iy = tile_y + j - 1;

					if (global_ix < 0) global_ix += n_tiles_x;
					else if (global_ix >= n_tiles_x) global_ix -= n_tiles_x;

					if(np[i + j * 3] > 0)
					{
						#pragma acc atomic
						np_per_tile[global_ix + global_iy * n_tiles_x] += np[i + j * 3];
					}
				}
			}
		}
	}

	// Add the incoming particles to the histogram
	for(int n = 0; n < 3; n++)
	{
		if(incoming_part[n].enable_vector)
		{
			int size_temp = incoming_part[n].size;;

			#pragma acc parallel loop
			for(int k = 0; k < size_temp; k++)
			{
				int ix = incoming_part[n].ix[k] / TILE_SIZE;
				int iy = (incoming_part[n].iy[k] - offset_region) / TILE_SIZE;
				int target_tile = ix + iy * n_tiles_x;

				#pragma acc atomic
				np_per_tile[target_tile]++;
			}
		}
	}

	// Copy the histogram to calculate the new tile offset
	#pragma acc parallel loop
	for (int i = 0; i <= n_tiles; i++)
		if(i < n_tiles) tile_offset[i] = np_per_tile[i];
		else tile_offset[i] = 0;
}

// Calculate an histogram for the particles moving between tiles
#pragma oss task device(openacc) in(*part_vector) in(tile_offset[0: n_tiles]) out(np_leaving[0: n_tiles]) \
		label("Sort (GPU, Histogram Leaving Part)")
void histogram_moving_particles(t_particle_vector *part_vector, int *restrict tile_offset,
		int *restrict np_leaving, const int n_tiles, const int n_tiles_x, const int offset_region,
		const int old_size, const int device)
{
#ifdef MANUAL_GPU_SETUP
	acc_set_device_num(device, DEVICE_TYPE);
#endif

	#pragma acc parallel loop gang
	for(int tile_idx = 0; tile_idx < n_tiles; tile_idx++)
	{
		const int begin = tile_offset[tile_idx];
		const int end = tile_offset[tile_idx + 1];
		int leaving_count = 0;

		#pragma acc loop vector reduction(+ : leaving_count)
		for (int k = begin; k < end; k++)
		{
			if(k >= old_size) part_vector->invalid[k] = true;

			int ix = part_vector->ix[k] / TILE_SIZE;
			int iy = (part_vector->iy[k] - offset_region) / TILE_SIZE;
			bool is_invalid = part_vector->invalid[k];
			int target_tile = ix + iy * n_tiles_x;

			if (is_invalid || target_tile != tile_idx) leaving_count++;
		}

		np_leaving[tile_idx] = leaving_count;
	}
}

// Identify the particles in the wrong tile and then generate a sorted list for them
// source idx - particle's current position / target idx - particle's new position
#pragma oss task device(openacc) in(*part_vector) in(tile_offset[0; n_tiles_x * n_tiles_y]) \
	in(mv_part_offset[0; n_tiles_x * n_tiles_y]) out(counter[0; n_tiles_x * n_tiles_y]) \
	out(source_idx[0; sorting_size]) out(target_idx[0; sorting_size]) label("Sort (GPU, Sorted Index)")
void calculate_sorted_idx(t_particle_vector *part_vector, int *restrict tile_offset,
		int *restrict source_idx, int *restrict target_idx, int *restrict counter,
		int *restrict mv_part_offset, const int n_tiles_y, const int n_tiles_x, const int old_size,
		const int offset_region, const int sorting_size, const int device)
{
	const int n_tiles = n_tiles_x * n_tiles_y;
	const int size = part_vector->size;

#ifdef MANUAL_GPU_SETUP
	acc_set_device_num(device, DEVICE_TYPE);
#endif

	#pragma acc parallel loop
	for(int i = 0; i < sorting_size; i++)
		source_idx[i] = -1;

	#pragma acc parallel loop
	for (int i = 0; i < n_tiles; i++)
		counter[i] = mv_part_offset[i];

	// Determine which particles are in the wrong tile
	#pragma acc parallel loop gang collapse(2)
	for(int tile_y = 0; tile_y < n_tiles_y; tile_y++)
	{
		for(int tile_x = 0; tile_x < n_tiles_x; tile_x++)
		{
			const int tile_idx = tile_x + tile_y * n_tiles_x;
			const int begin = tile_offset[tile_idx];
			const int end = tile_offset[tile_idx + 1];
			int offset = mv_part_offset[tile_idx];

			int right_counter = 0; // Count the particles moving to the right tile

			#pragma acc loop vector reduction(+ : right_counter)
			for (int k = begin; k < end; k++)
			{
				int idx;

				int ix = part_vector->ix[k] / TILE_SIZE;
				int iy = (part_vector->iy[k] - offset_region) / TILE_SIZE;
				bool is_invalid = part_vector->invalid[k];
				int target_tile = ix + iy * n_tiles_x;

				if (is_invalid || target_tile != tile_idx)
				{
					#pragma acc atomic capture
					idx = offset++;

					target_idx[idx] = k;
				}

				if(!is_invalid && target_tile == tile_idx + 1) right_counter++;
			}

			if(tile_x < n_tiles_x - 1)
			{
				#pragma acc atomic
				counter[tile_idx + 1] += right_counter; // Create a space for incoming particles from the current tile to its right tile
			}
		}
	}

	// Generate a sorted list for the particles in the wrong tile
	#pragma acc parallel loop gang collapse(2)
	for(int tile_y = 0; tile_y < n_tiles_y; tile_y++)
	{
		for(int tile_x = 0; tile_x < n_tiles_x; tile_x++)
		{
			const int tile_idx = tile_x + tile_y * n_tiles_x;
			const int begin = mv_part_offset[tile_idx];
			const int end = mv_part_offset[tile_idx + 1];

			int left_counter = begin - 1; // Local counter for the particle going to the left tile
			int right_counter = end; // Local counter for the particle going to the right tile

			#pragma acc loop vector
			for (int k = begin; k < end; k++)
			{
				int idx;
				int source = target_idx[k];
				int ix = part_vector->ix[source] / TILE_SIZE;
				int iy = (part_vector->iy[source] - offset_region) / TILE_SIZE;
				bool is_invalid = part_vector->invalid[source];

				int target_tile = ix + iy * n_tiles_x;

				if (!is_invalid)
				{
					if (tile_x > 0 && target_tile == tile_idx - 1)
					{
						#pragma acc atomic capture
						idx = left_counter--;
					}
					else if (tile_x < n_tiles_x - 1 && target_tile == tile_idx + 1)
					{
						#pragma acc atomic capture
						idx = right_counter++;
					}else
					{
						#pragma acc atomic capture
						idx = counter[target_tile]++;
					}

					source_idx[idx] = source;
				}
			}
		}
	}

	// If the vector has shrink, add the valid particles outside the vector new size
	if(size < old_size)
	{
		#pragma acc parallel loop
		for(int k = size; k < old_size; k++)
		{
			int idx;
			int ix = part_vector->ix[k] / TILE_SIZE;
			int iy = (part_vector->iy[k] - offset_region) / TILE_SIZE;
			bool is_invalid = part_vector->invalid[k];

			int target_tile = ix + iy * n_tiles_x;

			if (!is_invalid)
			{
				#pragma acc atomic capture
				idx = counter[target_tile]++;

				source_idx[idx] = k;
			}
		}
	}
}

// Merge the temporary vectors for the incoming particle into the main vector
#pragma oss task device(openacc) inout(*part_vector) inout(incoming_part[0:1]) \
	inout(target_idx[0; sorting_size]) inout(counter[0; n_tiles]) label("Sort (GPU, Merge Vectors)")
void merge_particles_buffers(t_particle_vector *part_vector, t_particle_vector *incoming_part,
		int *restrict counter, int *restrict target_idx, const int n_tiles_x, const int n_tiles,
		const int offset_region, const int sorting_size, const int device)
{
#ifdef MANUAL_GPU_SETUP
	acc_set_device_num(device, DEVICE_TYPE);
#endif

	for(int n = 0; n < 3; n++)
	{
		if(incoming_part[n].enable_vector)
		{
			int size_temp = incoming_part[n].size;;

			#pragma acc parallel loop
			for(int k = 0; k < size_temp; k++)
			{
				int idx;
				int ix = incoming_part[n].ix[k] / TILE_SIZE;
				int iy = (incoming_part[n].iy[k] - offset_region) / TILE_SIZE;
				int target_tile = ix + iy * n_tiles_x;

				#pragma acc atomic capture
				idx = counter[target_tile]++;

				int target = target_idx[idx];

				part_vector->ix[target] = incoming_part[n].ix[k];
				part_vector->iy[target] = incoming_part[n].iy[k];
				part_vector->x[target] = incoming_part[n].x[k];
				part_vector->y[target] = incoming_part[n].y[k];
				part_vector->ux[target] = incoming_part[n].ux[k];
				part_vector->uy[target] = incoming_part[n].uy[k];
				part_vector->uz[target] = incoming_part[n].uz[k];
				part_vector->invalid[target] = false;
			}

			incoming_part[n].size = 0;
		}
	}
}

// Reset the invalid mark from the particles
#pragma oss task device(openacc) inout(*invalid_mark) inout(source_idx[0; sorting_size]) \
		inout(target_idx[0; sorting_size]) label("Sort (GPU, Reset Invalid)")
void reset_invalid_part(bool *restrict invalid_mark, int *restrict source_idx,
        int *restrict target_idx, const int sorting_size, const int device)
{
#ifdef MANUAL_GPU_SETUP
	acc_set_device_num(device, DEVICE_TYPE);
#endif

	#pragma acc parallel loop
	for (int i = 0; i < sorting_size; i++)
		if (source_idx[i] >= 0) invalid_mark[target_idx[i]] = false;
}

#pragma oss task label("Sort (GPU, Sort Particles)") inout(*part_vector) inout(incoming_part[0:1]) \
	in(tile_offset[0: n_tiles_x * n_tiles_y]) in(mv_part_offset[0: n_tiles_x * n_tiles_y])
void spec_sort_particles(t_particle_vector *part_vector, t_particle_vector incoming_part[3],
		int *tile_offset, int *mv_part_offset, const int n_tiles_x, const int n_tiles_y,
		const int offset_region, const int old_size, const int device)
{
	int n_tiles = n_tiles_x * n_tiles_y;
	int sorting_size = mv_part_offset[n_tiles];
	int *restrict source_idx = malloc(sorting_size * sizeof(int));
	int *restrict target_idx = malloc(sorting_size * sizeof(int));
	int *restrict counter = malloc(n_tiles * sizeof(int));

	int *restrict temp_int = malloc(sorting_size * sizeof(int));
	float *restrict temp_float = malloc(sorting_size * sizeof(float));

	part_vector->size = tile_offset[n_tiles];

	//Generate a sorted list
	calculate_sorted_idx(part_vector, tile_offset, source_idx, target_idx, counter,
			mv_part_offset, n_tiles_y, n_tiles_x, old_size, offset_region, sorting_size, device);

	// Sort the main vector
	spec_move_vector_int(part_vector->ix, temp_int, source_idx, target_idx, sorting_size, device);
	spec_move_vector_int(part_vector->iy, temp_int, source_idx, target_idx, sorting_size, device);
	spec_move_vector_float(part_vector->x, temp_float, source_idx, target_idx, sorting_size, device);
	spec_move_vector_float(part_vector->y, temp_float, source_idx, target_idx, sorting_size, device);
	spec_move_vector_float(part_vector->ux, temp_float, source_idx, target_idx, sorting_size, device);
	spec_move_vector_float(part_vector->uy, temp_float, source_idx, target_idx, sorting_size, device);
	spec_move_vector_float(part_vector->uz, temp_float, source_idx, target_idx, sorting_size, device);
	reset_invalid_part(part_vector->invalid, source_idx, target_idx, sorting_size, device);

	// Merge all the temporary vectors into the main vector
	merge_particles_buffers(part_vector, incoming_part, counter, target_idx,
			n_tiles_x, n_tiles, offset_region, sorting_size, device);

	// Free temporary vectors
	#pragma oss task in(source_idx[0; sorting_size]) in(target_idx[0; sorting_size]) \
	in(counter[0; n_tiles]) in(temp_float[0; sorting_size]) in(temp_int[0; sorting_size])
	{
		free(temp_float);
		free(temp_int);
		free(counter);
		free(target_idx);
		free(source_idx);
	}
}

void spec_sort_openacc(t_species *spec, const int limits_y[2], const int device)
{
	const int n_tiles = spec->n_tiles_x * spec->n_tiles_y;
	spec->mv_part_offset[n_tiles] = 0;

	int *restrict np_per_tile = malloc(n_tiles * sizeof(int));

	int old_size = spec->main_vector.size;
	int np_inj = 0;
	for (int i = 0; i < 3; i++)
		if (spec->incoming_part[i].enable_vector) np_inj += spec->incoming_part[i].size;

	// Check if buffer is large enough and if not reallocate
	if (spec->main_vector.size + np_inj > spec->main_vector.size_max)
	{
		const int new_size = ((spec->main_vector.size_max + np_inj) / 1024 + 1) * 1024;
		part_vector_realloc(&spec->main_vector, new_size);
	}

#ifdef ENABLE_PREFETCH
	cudaMemPrefetchAsync(spec->tile_offset, (n_tiles + 1) * sizeof(int), device, NULL);
	spec_prefetch_openacc(&spec->main_vector, device);

	for(int i = 0; i < 3; i++)
		if(spec->incoming_part[i].enable_vector) spec_prefetch_openacc(&spec->incoming_part[i], device);
#endif

	// Calculate the new offset (in the particle vector) for the tiles
	histogram_np_per_tile(&spec->main_vector, spec->tile_offset, spec->incoming_part, np_per_tile,
	        spec->n_tiles_y, spec->n_tiles_x, limits_y[0], device);
	prefix_sum_openacc(spec->tile_offset, n_tiles + 1, device);

	#pragma oss task inout(spec->tile_offset[0; n_tiles])
	free(np_per_tile);

	// Calculate the offset for the moving particles between tiles
	histogram_moving_particles(&spec->main_vector, spec->tile_offset, spec->mv_part_offset, n_tiles,
	        spec->n_tiles_x, limits_y[0], old_size, device);
	prefix_sum_openacc(spec->mv_part_offset, n_tiles + 1, device);

	spec_sort_particles(&spec->main_vector, spec->incoming_part, spec->tile_offset,
			spec->mv_part_offset, spec->n_tiles_x, spec->n_tiles_y, limits_y[0],
			old_size, device);
}
