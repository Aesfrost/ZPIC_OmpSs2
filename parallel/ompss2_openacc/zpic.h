/*
 *  zpic.h
 *  zpic
 *
 *  Created by Ricardo Fonseca on 12/8/10.
 *  Copyright 2010 Centro de Física dos Plasmas. All rights reserved.
 *
 */

#ifndef __ZPIC__
#define __ZPIC__

#include <openacc.h>
#define DEVICE_TYPE acc_device_nvidia

typedef float t_fld;
typedef float t_part_data;

typedef struct {
	t_fld x, y, z;
} t_vfld;

typedef struct {
	int x, y;
} t_integer2;

typedef struct {
	t_part_data x, y;
} t_float2;

typedef struct {
	t_part_data x, y, z;
} t_float3;

/* ANSI C does not define math constants */

#ifndef M_PI
#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#endif

#ifndef M_PI_2
#define M_PI_2      1.57079632679489661923132169163975144   /* pi/2           */
#endif

#ifndef M_PI_4
#define M_PI_4      0.785398163397448309615660845819875721  /* pi/4           */
#endif

#endif
