#include "task_management.h"

#ifdef ENABLE_TASKING

typedef struct {
	void *context;
	gaspi_segment_id_t segm_id;
	int notif_id[8];
	bool is_blocked;
} t_comm_task;

static t_comm_task _blocked_tasks[MAX_BLOCKED_TASKS];
static unsigned int _blocked_tasks_count = 0;

static bool _is_finished;
static bool _enable_pooling_service;

// Checks periodically if the notifications from a communication task have arrived
// When this happens, resume the communication task
static void pooling_service(void *args)
{
	bool is_enabled;
	int segm_id, notif_id;

	do
	{
		for (int task_id = 0; task_id < MAX_BLOCKED_TASKS; task_id++)
		{
			if(_blocked_tasks[task_id].is_blocked)
			{
				segm_id = _blocked_tasks[task_id].segm_id;
				bool received_all = true;

				for (int i = 0; i < 8; ++i)
				{
					notif_id = _blocked_tasks[task_id].notif_id[i];
					if(notif_id >= 0)
						received_all &= gaspi_notify_test(segm_id, notif_id);
				}

				if(received_all)
				{
					_blocked_tasks[task_id].is_blocked = false;
					nanos6_unblock_task(_blocked_tasks[task_id].context);
				}
			}
		}

		nanos6_wait_for(100);

		#pragma omp atomic read
		is_enabled = _enable_pooling_service;

	}while(is_enabled);
}

void complete_service(void *args)
{
	#pragma omp atomic write
	_is_finished = true;
}

// Init the task management mechanism
void init_task_management()
{
	for (int task_id = 0; task_id < MAX_BLOCKED_TASKS; task_id++)
		_blocked_tasks[task_id].is_blocked = false;

	_is_finished = false;
	_enable_pooling_service = true;

	nanos6_spawn_function(pooling_service, NULL, complete_service, NULL, "CommTask Management");
}

// Delete the task management mechanism
void delete_task_management()
{
	_enable_pooling_service = false;

	bool is_completed;

	do
	{
		#pragma omp atomic read
		is_completed = _is_finished;
	} while(!is_completed);
}

// Block a communication task until all notifications arrived
void block_comm_task(const gaspi_segment_id_t segm_id, const int notif_ids[8])
{
	int id;
	t_comm_task task;

	#pragma omp atomic capture
	id = _blocked_tasks_count++;
	id = id % MAX_BLOCKED_TASKS;

	task.context = nanos6_get_current_blocking_context();
	task.is_blocked = true;
	task.segm_id = segm_id;

	for (int i = 0; i < NUM_ADJ_PART; ++i)
		task.notif_id[i] = notif_ids[i];

	if(!_blocked_tasks[id].is_blocked)
	{
		_blocked_tasks[id] = task;
		nanos6_block_current_task(task.context);
	}else
	{
		gaspi_notification_id_t notif;
		for (int i = 0; i < NUM_ADJ_PART; ++i)
			if(notif_ids[i] >= 0)
				CHECK_GASPI_ERROR(gaspi_notify_waitsome(segm_id, notif_ids[i], 1, &notif, GASPI_BLOCK));
	}
}

#endif
