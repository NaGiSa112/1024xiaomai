# -*- coding: utf-8 -*-
"""Sandbox compatibility shim -- import this BEFORE ultralytics.

The DSH sandbox denies named-pipe creation, so `multiprocessing.Pool`
(and `multiprocessing.pool.ThreadPool`, which still builds a SimpleQueue)
raises PermissionError: [WinError 5]. Ultralytics uses both for label caching
and DataLoader workers, so they are replaced with serial equivalents here.

Effect: label-cache scans and data loading run in-process. Slightly slower,
identical results. Keep `workers=0` as the default in this project.
"""
import multiprocessing.pool as _mp_pool


class _SerialPool:
    """Drop-in for the subset of multiprocessing.Pool ultralytics actually uses."""

    def __init__(self, *args, **kwargs):
        self._closed = False

    # context manager
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # work
    def imap(self, func, iterable, chunksize=None):
        for item in iterable:
            yield func(item)

    def imap_unordered(self, func, iterable, chunksize=None):
        for item in iterable:
            yield func(item)

    def map(self, func, iterable, chunksize=None):
        return [func(item) for item in iterable]

    def starmap(self, func, iterable, chunksize=None):
        return [func(*item) for item in iterable]

    # lifecycle no-ops
    def close(self):
        self._closed = True

    def join(self):
        return None

    def terminate(self):
        self._closed = True


_mp_pool.ThreadPool = _SerialPool      # ultralytics.data.dataset imports this name
_mp_pool.Pool = _SerialPool            # safety net for direct Pool users
