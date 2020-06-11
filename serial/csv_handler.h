/*********************************************************************************************
 ZPIC
 csv_handler.h

 Created by Nicolas Guidotti on 11/06/2020

 Copyright 2010 Centro de Física dos Plasmas. All rights reserved.

 *********************************************************************************************/

#ifndef SERIAL_CSV_HANDLER_H_
#define SERIAL_CSV_HANDLER_H_

#include "zpic.h"

// Write the data into a .csv file
void save_data_csv(t_fld *grid, unsigned int sizeX, unsigned int sizeY, const char filename[128], const char sim_name[64]);

#endif /* SERIAL_CSV_HANDLER_H_ */
