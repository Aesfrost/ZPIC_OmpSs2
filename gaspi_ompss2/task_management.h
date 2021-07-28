#ifndef _MANAGEMENT_H_
#define _TASK_MANAGEMENT_H_

#ifdef ENABLE_TASKING
#include <stdbool.h>
#include <nanos6.h>
#include <GASPI.h>

#include "utilities.h"

#define MAX_BLOCKED_TASKS 64

void init_task_management();
void delete_task_management();
void block_comm_task(const gaspi_segment_id_t segm_id, const int notif_ids[8]);

#endif
#endif /* _TASK_MANAGEMENT_H_ */
