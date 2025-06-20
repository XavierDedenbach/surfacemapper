import tempfile
import os
from contextlib import contextmanager

@contextmanager
def safe_temp_file(suffix="", prefix="tmp", dir=None, delete=True):
    """
    Context manager for safe temp file creation and cleanup.
    Yields the file path. Deletes the file on exit if delete=True.
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
    os.close(fd)
    try:
        yield path
    finally:
        if delete and os.path.exists(path):
            os.remove(path) 