#include "task_management.h"

#ifdef ENABLE_TASKING

typedef struct {
	void *context;
	MPI_Request *requests;
	int num_requests;
	bool is_blocked;
} t_comm_task;

static t_comm_task _blocked_tasks[MAX_BLOCKED_TASKS];
static unsigned int _blocked_tasks_count = 0;

// Checks periodically if the notifications from a communication task have arrived
// When this happens, resume the communication task
static int poolingService(void *data)
{
	t_comm_task task;
	bool is_blocked;

	for (int task_id = 0; task_id < MAX_BLOCKED_TASKS; task_id++)
	{
		#pragma omp atomic read
		is_blocked = _blocked_tasks[task_id].is_blocked;

		if(is_blocked)
		{
			task = _blocked_tasks[task_id];

			int received = 0;
			CHECK_MPI_ERROR(MPI_Testall(task.num_requests, task.requests, &received, MPI_STATUSES_IGNORE));

			if(received)
			{
				_blocked_tasks[task_id].is_blocked = false;
				nanos6_unblock_task(task.context);
			}
		}
	}

	return 0;
}

// Init the task management mechanism
void init_task_management()
{
	for (int task_id = 0; task_id < MAX_BLOCKED_TASKS; task_id++)
		_blocked_tasks[task_id].is_blocked = false;
	nanos6_register_polling_service("CommTaskManagement", poolingService, NULL);
}

// Delete the task management mechanism
void delete_task_management()
{
	nanos6_unregister_polling_service("CommTaskManagement", poolingService, NULL);
}

// Block a communication task until all notifications arrived
void block_comm_task(MPI_Request *requests, const int num_requests)
{
	int id;

	#pragma omp atomic capture
	id = _blocked_tasks_count++;
	id = id % MAX_BLOCKED_TASKS;

	if(!_blocked_tasks[id].is_blocked)
	{
		_blocked_tasks[id].requests = requests;
		_blocked_tasks[id].num_requests = num_requests;
		_blocked_tasks[id].context = nanos6_get_current_blocking_context();

		#pragma omp atomic write
		_blocked_tasks[id].is_blocked = true;

		nanos6_block_current_task(_blocked_tasks[id].context);
	}else
	{
		CHECK_MPI_ERROR(MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE));
	}
}

#endif
