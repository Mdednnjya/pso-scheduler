import asyncio
import logging
from typing import Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """Manager untuk background tasks dengan monitoring"""

    def __init__(self):
        self.active_tasks = {}
        self.completed_tasks = {}

    async def add_task(self, task_id: str, coro: Callable, *args, **kwargs):
        """Add background task dengan tracking"""
        try:
            # Create task
            task = asyncio.create_task(coro(*args, **kwargs))
            self.active_tasks[task_id] = {
                'task': task,
                'started_at': asyncio.get_event_loop().time(),
                'status': 'running'
            }

            # Monitor completion
            asyncio.create_task(self._monitor_task(task_id, task))

            logger.info(f"📋 Background task started: {task_id}")
            return task

        except Exception as e:
            logger.error(f"❌ Failed to start background task {task_id}: {e}")
            raise

    async def _monitor_task(self, task_id: str, task: asyncio.Task):
        """Monitor task completion"""
        try:
            result = await task

            # Move to completed
            if task_id in self.active_tasks:
                task_info = self.active_tasks.pop(task_id)
                self.completed_tasks[task_id] = {
                    'status': 'completed',
                    'started_at': task_info['started_at'],
                    'completed_at': asyncio.get_event_loop().time(),
                    'result': result
                }

            logger.info(f"✅ Background task completed: {task_id}")

        except asyncio.CancelledError:
            logger.warning(f"⚠️ Background task cancelled: {task_id}")
            self._mark_task_cancelled(task_id)

        except Exception as e:
            logger.error(f"❌ Background task failed {task_id}: {e}")
            self._mark_task_failed(task_id, str(e))

    def _mark_task_cancelled(self, task_id: str):
        """Mark task as cancelled"""
        if task_id in self.active_tasks:
            task_info = self.active_tasks.pop(task_id)
            self.completed_tasks[task_id] = {
                'status': 'cancelled',
                'started_at': task_info['started_at'],
                'completed_at': asyncio.get_event_loop().time()
            }

    def _mark_task_failed(self, task_id: str, error: str):
        """Mark task as failed"""
        if task_id in self.active_tasks:
            task_info = self.active_tasks.pop(task_id)
            self.completed_tasks[task_id] = {
                'status': 'failed',
                'started_at': task_info['started_at'],
                'completed_at': asyncio.get_event_loop().time(),
                'error': error
            }

    def get_task_status(self, task_id: str) -> dict:
        """Get status of specific task"""
        if task_id in self.active_tasks:
            return {
                'status': 'running',
                'started_at': self.active_tasks[task_id]['started_at']
            }
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        else:
            return {'status': 'not_found'}

    def get_active_tasks_count(self) -> int:
        """Get number of active tasks"""
        return len(self.active_tasks)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel specific task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]['task']
            task.cancel()
            logger.info(f"🚫 Task cancelled: {task_id}")
            return True
        return False

    async def cancel_all_tasks(self):
        """Cancel all active tasks"""
        for task_id in list(self.active_tasks.keys()):
            await self.cancel_task(task_id)

        logger.info("🧹 All background tasks cancelled")


# Global task manager instance
task_manager = BackgroundTaskManager()


def async_task(task_name: str = None):
    """Decorator untuk async background tasks"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate task name if not provided
            if task_name is None:
                name = f"{func.__name__}_{id(func)}"
            else:
                name = task_name

            # Execute with monitoring
            try:
                result = await func(*args, **kwargs)
                logger.info(f"🎯 Async task completed: {name}")
                return result
            except Exception as e:
                logger.error(f"💥 Async task failed {name}: {e}")
                raise

        return wrapper

    return decorator


# Utility functions for task management
async def wait_for_task_completion(task_id: str, timeout: float = 60.0) -> dict:
    """Wait for task completion with timeout"""
    start_time = asyncio.get_event_loop().time()

    while asyncio.get_event_loop().time() - start_time < timeout:
        status = task_manager.get_task_status(task_id)

        if status['status'] in ['completed', 'failed', 'cancelled']:
            return status

        await asyncio.sleep(0.5)  # Check every 500ms

    # Timeout reached
    await task_manager.cancel_task(task_id)
    return {'status': 'timeout', 'message': f'Task timed out after {timeout}s'}


async def cleanup_old_tasks(max_age_seconds: int = 3600):
    """Cleanup old completed tasks"""
    current_time = asyncio.get_event_loop().time()
    to_remove = []

    for task_id, task_info in task_manager.completed_tasks.items():
        if current_time - task_info.get('completed_at', 0) > max_age_seconds:
            to_remove.append(task_id)

    for task_id in to_remove:
        del task_manager.completed_tasks[task_id]

    if to_remove:
        logger.info(f"🧹 Cleaned up {len(to_remove)} old tasks")