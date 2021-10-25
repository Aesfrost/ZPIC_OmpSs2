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

static bool _is_finished;
static bool _enable_pooling_service;

// Checks periodically if the notifications from a communication task have arrived
// When this happens, resume the communication task
void pooling_service(void *args)
{
	t_comm_task task;
	bool is_blocked;
	bool is_enabled;

	do
	{
		for (int task_id = 0; task_id < MAX_BLOCKED_TASKS; task_id++)
		{
			#pragma omp atomic read
			is_blocked = _blocked_tasks[task_id].is_blocked;

			if(is_blocked)
			{
				task = _blocked_tasks[task_id];

				int received = 0;
				CHECK_MPI_ERROR(MPI_Testall(task.num_requests, task.requests,
											&received, MPI_STATUSES_IGNORE));

				if(received)
				{
					_blocked_tasks[task_id].is_blocked = false;
					nanos6_unblock_task(task.context);
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
