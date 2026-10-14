"""Scoped numerical thread control for model execution."""

from contextlib import contextmanager
from threading import RLock


_TORCH_LOCK = RLock()


@contextmanager
def single_threaded_torch(n_threads=1):
    """Temporarily limit threads and restore PyTorch settings after execution.

    PyTorch thread settings are process-wide. The lock prevents overlapping
    calls to this helper from changing each other's settings.
    """
    import torch
    from threadpoolctl import threadpool_limits

    with _TORCH_LOCK:
        previous_threads = torch.get_num_threads()
        with threadpool_limits(limits=n_threads):
            torch.set_num_threads(n_threads)
            try:
                yield
            finally:
                torch.set_num_threads(previous_threads)
