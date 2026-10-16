"""One numerical thread per process; parallel analyses use separate processes."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

# These are inherited by spawned workers and apply to libraries loaded later.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

_INITIALIZED_PID = None
_THREAD_LIMITER = None


def initialize_worker():
    """Configure the main process or a new worker once, before model execution.

    Also limit libraries that a notebook imported before benchmark. PyTorch's
    inter-operation setting can only be changed before parallel work starts;
    an existing kernel may therefore need a restart.
    """
    global _INITIALIZED_PID, _THREAD_LIMITER
    if _INITIALIZED_PID == os.getpid():
        return
    import torch
    from threadpoolctl import threadpool_limits

    if "numexpr" in sys.modules:
        sys.modules["numexpr"].set_num_threads(1)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise RuntimeError(
                "Restart the Python process or notebook kernel, then import benchmark "
                "before running model computations to enable one-thread execution."
            ) from error
    # Keep the limit active for the lifetime of this process.
    _THREAD_LIMITER = threadpool_limits(limits=1)
    _INITIALIZED_PID = os.getpid()


def run_logged_iteration(function, warnings_file, *args):
    """Run one process task with private logs; return its warnings to the parent.

    Files remain available even if the task fails. The parent appends completed
    tasks' messages to the main log, while only its progress bar stays visible.
    """
    from .utils import capture_output

    path = Path(warnings_file)
    with capture_output(path, path.parent) as (records, progress):
        result = function(*args)
    return result, records, str(path.with_suffix(".log"))


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
