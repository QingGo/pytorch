from __future__ import annotations

import enum
import functools
import io
import linecache
import os
import pickle
import sys
import threading
import time
import warnings
from multiprocessing.connection import Client
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any, TYPE_CHECKING

from torch._utils_internal import log_triton_builds


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._inductor.runtime.triton_heuristics import CachingAutotuner


def _reload_python_module(
    key: str, path: str, set_sys_modules: bool = True
) -> ModuleType:
    with open(path) as f:
        try:
            code = compile(f.read(), path, "exec", dont_inherit=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {path}\n{type(e).__name__}: {e}"
            ) from None
        mod = ModuleType(f"{__name__}.{key}")
        mod.__file__ = path
        mod.key = key  # type: ignore[attr-defined]
        exec(code, mod.__dict__, mod.__dict__)
        if set_sys_modules:
            sys.modules[mod.__name__] = mod
        return mod


@functools.cache
def _set_triton_ptxas_path() -> None:
    if os.environ.get("TRITON_PTXAS_PATH") is not None:
        return
    ptxas = Path(__file__).absolute().parents[1] / "bin" / "ptxas"
    if not ptxas.exists():
        return
    if ptxas.is_file() and os.access(ptxas, os.X_OK):
        os.environ["TRITON_PTXAS_PATH"] = str(ptxas)
    else:
        warnings.warn(f"{ptxas} exists but is not an executable")


def _set_triton_libdevice_path() -> None:
    """
    Use the CUDA toolkit's libdevice instead of Triton's bundled version.
    This ensures Triton's pow matches CUDA's powf for bitwise precision.
    Gated by config.eager_numerics.use_pytorch_libdevice.
    """
    from torch._inductor import config

    if not config.eager_numerics.use_pytorch_libdevice:
        return

    _set_triton_libdevice_path_impl()


def _set_triton_libdevice_path_impl() -> None:
    try:
        from triton import knobs
    except ImportError:
        return

    env_path = os.environ.get("TRITON_LIBDEVICE_PATH")
    if env_path is not None:
        knobs.nvidia.libdevice_path = env_path
        return

    if knobs.nvidia.libdevice_path is not None:
        return

    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            warnings.warn(
                "CUDA_HOME not set; using Triton's bundled libdevice which may "
                "cause minor precision differences in pow operations. "
                "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
                "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
                stacklevel=3,
            )
            return
        libdevice = Path(CUDA_HOME) / "nvvm" / "libdevice" / "libdevice.10.bc"
        if libdevice.is_file():
            knobs.nvidia.libdevice_path = str(libdevice)
            # Also set env var so subprocess compile workers inherit it
            os.environ["TRITON_LIBDEVICE_PATH"] = str(libdevice)
        else:
            warnings.warn(
                f"CUDA libdevice not found at {libdevice}; using Triton's bundled "
                "libdevice which may cause minor precision differences in pow operations. "
                "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
                "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
                stacklevel=3,
            )
    except ImportError:
        warnings.warn(
            "torch.utils.cpp_extension not available; using Triton's bundled "
            "libdevice which may cause minor precision differences in pow operations. "
            "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
            "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
            stacklevel=3,
        )


def _worker_compile_triton(
    load_kernel: Callable[[], CachingAutotuner],
    extra_env: dict[str, str],
    extra_config: dict[str, Any],
    streaming_address: str | None = None,
    streaming_authkey: bytes | None = None,
) -> tuple[CachingAutotuner | None, int]:
    """Worker entry point for ``AsyncCompile.triton``. Two flows:

    - Streaming (``streaming_address`` non-None): send the kernel first,
      then stream each ``CompileResult``. Returns ``(None, elapsed_us)``.
      ``streaming_authkey`` authenticates the connection back to the parent.
    - Blocking (None): ``precompile(warm_cache_only=True)`` runs every
      config, kernel is pickled back with ``compile_results`` populated.
      Returns ``(kernel, elapsed_us)``.
    """
    _set_triton_ptxas_path()
    os.environ.update(extra_env)
    # Set libdevice path if passed via env from main process
    libdevice_path = extra_env.get("TRITON_LIBDEVICE_PATH")
    if libdevice_path:
        try:
            from triton import knobs

            knobs.nvidia.libdevice_path = libdevice_path
        except ImportError:
            pass
    from torch._inductor import config

    with config.patch(extra_config):
        fail = None
        try:
            start_ns = time.time_ns()
            kernel = load_kernel()
            if streaming_address is not None:
                assert streaming_authkey is not None
                _stream_compile_triton(
                    kernel, streaming_address, streaming_authkey
                )
                elapsed_ns = time.time_ns() - start_ns
                # Kernel was streamed back already; nothing left to
                # return via the future payload.
                linecache.clearcache()
                return None, elapsed_ns // 1000
            kernel.precompile(warm_cache_only=True)
            elapsed_ns = time.time_ns() - start_ns
            kernel.prepare_for_pickle()
            # We can release this memory in the compile subprocesses:
            linecache.clearcache()
            return kernel, elapsed_ns // 1000
        except Exception as e:
            fail = str(e)
            raise
        finally:
            log_triton_builds(fail=fail)


_DYN_KERNEL_MODULE_PREFIX = "torch._inductor.runtime.compile_tasks."

# RLock's type isn't directly importable; capture once via type(RLock()).
# _streaming_persistent_id checks this on every traversed pickle object.
_RLOCK_TYPE = type(threading.RLock())


class _StreamingSentinel(enum.Enum):
    """Wire sentinels for the worker->parent streaming connection.
    Enum members round-trip as ``is``-identical singletons."""

    SKIP = enum.auto()


def _streaming_persistent_id(obj: Any) -> object | None:
    """Substitute ``_StreamingSentinel.SKIP`` for things that don't
    survive pickle to the parent. All resolve to ``None`` on the
    parent; downstream consumers either don't read them or handle
    ``None`` defensively.

    - ``RLock``: pickle refuses these outright.
    - ``ModuleType``: the dyn kernel module isn't in the parent's
      ``sys.modules``; uniform substitution is the simplest correct policy.
    - Raw functions in a dyn kernel module (e.g. ``@triton.jit`` bodies):
      pickle-by-name fails on the parent. Restricted to ``FunctionType``
      so we don't catch ``JITFunction``, which inherits the same
      ``__module__`` but pickles fine via its own class.
    """
    if isinstance(obj, _RLOCK_TYPE):
        return _StreamingSentinel.SKIP
    if isinstance(obj, ModuleType):
        return _StreamingSentinel.SKIP
    if isinstance(obj, FunctionType):
        mod = obj.__module__
        if isinstance(mod, str) and mod.startswith(_DYN_KERNEL_MODULE_PREFIX):
            return _StreamingSentinel.SKIP
    return None


def _streaming_persistent_load(pid: object) -> object:
    """Resolve ids from ``_streaming_persistent_id`` to ``None``;
    raise on anything else (wire format divergence)."""
    if pid is _StreamingSentinel.SKIP:
        return None
    raise pickle.UnpicklingError(f"unsupported persistent id: {pid!r}")


class _StreamingPickler(pickle.Pickler):
    """Pickler honoring ``_streaming_persistent_id`` substitutions."""

    persistent_id = staticmethod(_streaming_persistent_id)


class _StreamingUnpickler(pickle.Unpickler):
    """Parent-side companion to ``_StreamingPickler``."""

    persistent_load = staticmethod(_streaming_persistent_load)


def _streaming_send(conn: Any, obj: Any) -> None:
    """Pickle ``obj`` via ``_StreamingPickler`` and ``send_bytes`` it.
    Bypasses ``conn.send``, which would use ``ForkingPickler``."""
    buf = io.BytesIO()
    _StreamingPickler(buf, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)
    conn.send_bytes(buf.getvalue())


def _stream_compile_triton(
    kernel: CachingAutotuner, streaming_address: str, streaming_authkey: bytes
) -> None:
    """Connect to the parent's per-kernel AF_UNIX listener at
    ``streaming_address``, authenticate via ``streaming_authkey``, send the
    kernel as the first message (so the parent can dispatch before any compile
    finishes), then stream each ``CompileResult`` as it lands.
    ``_StreamingPickler`` substitutes unpicklable bits per-send so we never
    mutate the live JITFunction (other in-flight compiles in this worker still
    need it).
    """
    conn = Client(streaming_address, authkey=streaming_authkey)
    try:
        # No in-flight compiles yet; mutating kernel is safe.
        old_values = kernel.prepare_for_pickle()
        try:
            _streaming_send(conn, kernel)
        finally:
            kernel.restore_after_unpickle(old_values)

        for item in kernel._iter_compile_results(parallel=True):
            _streaming_send(conn, item)
    finally:
        conn.close()
