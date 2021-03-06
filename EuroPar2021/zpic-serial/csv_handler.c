/*********************************************************************************************
 ZPIC
 csv_handler.c

 Created by Nicolas Guidotti on 11/06/2020

 Copyright 2010 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>
#include <stdbool.h>

#include "csv_handler.h"

void save_data_csv(t_fld *grid, unsigned int sizeX, unsigned int sizeY, const char filename[128],
		const char sim_name[64])
{
	static bool dir_exists = false;

	char fullpath[256];

	if(!dir_exists)
	{
		//Create the output directory if it doesn't exists
		strcpy(fullpath, "output");
		struct stat sb;
		if (stat(fullpath, &sb) == -1)
		{
			mkdir(fullpath, 0700);
		}

		strcat(fullpath, "/");
		strcat(fullpath, sim_name);
		if (stat(fullpath, &sb) == -1)
		{
			mkdir(fullpath, 0700);
		}
	}else
	{
		strcpy(fullpath, "output/");
		strcat(fullpath, sim_name);
	}

	strcat(fullpath, "/");
	strcat(fullpath, filename);

	FILE *file = fopen(fullpath, "wb+");

	if (file != NULL)
	{
		for (unsigned int j = 0; j < sizeY; j++)
		{
			for (unsigned int i = 0; i < sizeX - 1; i++)
			{
				fprintf(file, "%f;", grid[i + j * sizeX]);
			}
			fprintf(file, "%f\n", grid[(j + 1) * sizeX - 1]);
		}
	} else
	{
		printf("Couldn't open %s", filename);
		exit(1);
	}

	fclose(file);
}
