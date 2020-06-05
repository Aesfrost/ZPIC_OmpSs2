#ifndef SERIAL_CSV_HANDLER_H_
#define SERIAL_CSV_HANDLER_H_

#include "zpic.h"

void save_data_csv(t_fld *grid, unsigned int sizeX, unsigned int sizeY, const char filename[128], const char sim_name[64]);

#endif /* SERIAL_CSV_HANDLER_H_ */
