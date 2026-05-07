# mypy: allow-untyped-defs
from __future__ import annotations

import atexit
import functools
import io
import itertools
import json
import logging
import multiprocessing
import os
import queue as _queue
import re
import shutil
import socket
import sys
import tempfile
import threading
from concurrent.futures import (
    Future,
    InvalidStateError,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from functools import partial
from multiprocessing.connection import answer_challenge, Connection, deliver_challenge
from time import time, time_ns
from typing import Any, cast, TYPE_CHECKING

import torch
from torch._dynamo.device_interface import get_registered_device_interfaces
from torch._dynamo.utils import (
    counters,
    dynamo_timed,
    get_metrics_context,
    set_feature_use,
)
from torch._inductor import config
from torch._inductor.codecache import (
    _load_triton_kernel_from_source,
    code_hash,
    CodeCacheFuture,
    CppCodeCache,
    CppPythonBindingsCodeCache,
    CUDACodeCache,
    HalideCodeCache,
    LambdaFuture,
    ROCmCodeCache,
    StaticAutotunerFuture,
    torch_key,
    XPUCodeCache,
)
from torch._inductor.compile_worker.subproc_pool import (
    AnyPool,
    SubprocException,
    SubprocPool,
)
from torch._inductor.compile_worker.tracked_process_pool import (
    TrackedProcessPoolExecutor,
)
from torch._inductor.compile_worker.utils import _async_compile_initializer
from torch._inductor.runtime.compile_tasks import (
    _set_triton_libdevice_path,
    _set_triton_ptxas_path,
    _StreamingUnpickler,
    _worker_compile_triton,
)
from torch._inductor.utils import clear_on_fresh_cache
from torch._inductor.virtualized import V
from torch._utils_internal import log_triton_builds
from torch.hub import _Faketqdm, tqdm
from torch.utils._ordered_set import OrderedSet
from torch.utils._triton import has_triton_package


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch._inductor.runtime.hints import HalideMeta
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

# timing metrics for time spent in the compilation
_cumulative_compile_time = 0.0
_t0: float | None = None

kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")

log = logging.getLogger(__name__)

_triton_kernel_metrics: dict[str, dict[str, Any]] | None = None

# EOF marker on ``launcher_q`` (single producer: ``_bg_drain_kernel``).
_LAUNCHER_END = object()


def _try_set_exception(fut: "Future[Any]", exc: BaseException) -> None:
    """Race-tolerant ``Future.set_exception``."""
    try:
        fut.set_exception(exc)
    except InvalidStateError:
        pass


def _try_set_result(fut: "Future[Any]", value: Any) -> None:
    """Race-tolerant ``Future.set_result``."""
    try:
        fut.set_result(value)
    except InvalidStateError:
        pass


size_hints_regex = re.compile(
    r"size_hints=(\{.*?\})",
)


@dataclass
class PipelineCachingAutotunerHandle:
    """Per-kernel state shared between ``_bg_drain_kernel`` (writer) and
    ``PipelineCachingAutotunerPlugin.pre_dispatch`` (reader). EOF on
    ``launcher_q`` is ``_LAUNCHER_END``. ``bg_drain_wait_ns`` is
    happens-before-published via ``drain_future`` resolution.
    """

    launcher_q: _queue.Queue[object]
    drain_future: Future[None]
    num_configs: int
    kernel_name: str
    bg_drain_wait_ns: int = 0


# One process-wide tmpdir for per-kernel AF_UNIX sockets, lazy-created.
_STREAMING_SOCK_DIR: str | None = None
_STREAMING_SOCK_DIR_LOCK = threading.Lock()

# Monotonic per-process suffix for sock filenames; next() is GIL-atomic, so
# names are guaranteed unique within a process. mkdtemp gives a fresh dir per
# process, so cross-process collisions are impossible.
_STREAMING_SOCK_COUNTER = itertools.count()

# 10 minutes. Bounds both accept() (worker startup) and recv_bytes() (per-
# config compile). Picked to be far above any realistic compile time so that
# only a wedged worker or hijacked listener trips it.
_STREAMING_TIMEOUT_S = 600.0


def _get_streaming_sock_dir() -> str:
    global _STREAMING_SOCK_DIR
    if _STREAMING_SOCK_DIR is None:
        with _STREAMING_SOCK_DIR_LOCK:
            if _STREAMING_SOCK_DIR is None:
                d = tempfile.mkdtemp(prefix="inductor_stream_")
                atexit.register(shutil.rmtree, d, ignore_errors=True)
                _STREAMING_SOCK_DIR = d
    return _STREAMING_SOCK_DIR


def _setup_streaming_listener(
    kernel_name: str,
) -> tuple[str, bytes, socket.socket]:
    """Bind a per-kernel AF_UNIX listener. Returns (sock_path, authkey, listen_sock).
    Worker connects to ``sock_path`` and authenticates with ``authkey``; parent
    ``accept()``s on ``listen_sock`` and runs the mp.connection challenge handshake.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", kernel_name)[:40]
    sock_path = os.path.join(
        _get_streaming_sock_dir(),
        f"{safe}_{next(_STREAMING_SOCK_COUNTER)}.sock",
    )
    listen_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listen_sock.bind(sock_path)
    # Don't trust the caller's umask: bind() inherits it. 0700 parent dir
    # already blocks cross-UID, but explicit chmod survives anyone changing
    # the parent path later.
    os.chmod(sock_path, 0o600)
    listen_sock.listen(1)
    listen_sock.settimeout(_STREAMING_TIMEOUT_S)
    authkey = os.urandom(32)
    return sock_path, authkey, listen_sock


def _bg_drain_kernel(
    kernel: Any,
    conn: Connection,
    handle: PipelineCachingAutotunerHandle,
    static_triton_bundle_key: str | None,
) -> None:
    """Per-kernel daemon: drives ``precompile_from_stream`` with results
    read off the worker's pipe, fanning launchers out to ``launcher_q``.
    """

    def _iter_results() -> "Iterable[Any]":
        # ``recv_bytes()`` blocks waiting on the worker; that wait is the
        # parent-side cost the plugin folds into compile_time_us.
        while True:
            tq0 = time_ns()
            if not conn.poll(_STREAMING_TIMEOUT_S):
                raise TimeoutError(
                    f"streaming worker for {handle.kernel_name} sent no data "
                    f"for {_STREAMING_TIMEOUT_S:.0f}s"
                )
            try:
                payload = conn.recv_bytes()
            except EOFError:
                return
            handle.bg_drain_wait_ns += time_ns() - tq0
            yield _StreamingUnpickler(io.BytesIO(payload)).load()

    try:
        kernel.precompile_from_stream(
            iter_results=_iter_results(),
            on_launcher=handle.launcher_q.put,
            static_triton_bundle_key=static_triton_bundle_key,
        )
    except BaseException as e:
        _try_set_exception(handle.drain_future, e)
        log.exception("background drain failed for %s", kernel.fn.__name__)
    else:
        _try_set_result(handle.drain_future, None)
    finally:
        handle.launcher_q.put(_LAUNCHER_END)
        try:
            conn.close()
        except OSError:
            pass


def _make_pipeline_caching_autotuner_plugin() -> Any:
    """Lazy factory: keeps runtime-side imports off the default-off path."""
    from torch._inductor.runtime.triton_heuristics import (
        CachingAutotunerPlugin,
        DEFER,
    )

    class PipelineCachingAutotunerPlugin(CachingAutotunerPlugin):
        """At first ``run()``: drain ``launcher_q`` (multi-config) or
        wait on ``drain_future`` (single-config), bench, finalize, and
        return ``DEFER`` so the standard dispatch path runs with
        ``self.launchers == [winner]``.
        """

        def pre_dispatch(
            self,
            autotuner: object,
            *args: object,
            stream: object,
            **kwargs: object,
        ) -> object:
            from torch._dynamo.device_interface import DeviceGuard
            from torch._inductor.runtime.triton_heuristics import (
                NoTritonConfigsError,
            )

            handle = autotuner._pipeline_caching_autotuner_handle
            if handle is None:
                return DEFER
            autotuner._pipeline_caching_autotuner_handle = None

            plugin_wait_ns = 0
            bench_ns = 0

            if handle.num_configs <= 1:
                # Nothing to compare against; just wait for finalization.
                tq0 = time_ns()
                handle.drain_future.result()  # re-raises drain exception
                plugin_wait_ns += time_ns() - tq0
            else:
                def _drain_launchers() -> "Iterable[Any]":
                    nonlocal plugin_wait_ns
                    while True:
                        tq0 = time_ns()
                        launcher = handle.launcher_q.get()
                        plugin_wait_ns += time_ns() - tq0
                        if launcher is _LAUNCHER_END:
                            return
                        yield launcher

                with DeviceGuard(
                    autotuner.get_device_interface(),
                    autotuner.triton_meta["device"],
                ):
                    timings, bench_ns = autotuner._bench_launchers(
                        _drain_launchers(), *args, **kwargs
                    )
                # Drain pushed its sentinel even on failure; surface now.
                handle.drain_future.result()

            if not autotuner.launchers and not autotuner.compile_results:
                raise NoTritonConfigsError(
                    f"Pipelined autotuner produced 0 results for "
                    f"{autotuner.fn.__name__}"
                )

            # parent_us = wall time the parent's threads spent blocked
            # waiting for compile data. Worker-subprocess time is
            # intentionally not measured here.
            parent_wait_ns = handle.bg_drain_wait_ns + plugin_wait_ns

            if handle.num_configs > 1 and timings:
                autotuner._finalize_autotune_winner(
                    timings,
                    autotune_time_taken_ns=parent_wait_ns + bench_ns,
                )

            _emit_triton_kernel_compile_metric(
                handle.kernel_name,
                parent_wait_ns // 1000,
                autotune_cache_info=autotuner.autotune_cache_info,
            )
            return DEFER

    return PipelineCachingAutotunerPlugin()


def pre_fork_setup():
    """
    Setup that must be done prior to forking with a process pool.
    """
    # ensure properties have been calculated before processes
    # are forked
    caching_device_properties()

    # Computing the triton key can be slow. If we call it before fork,
    # it will be cached for the forked subprocesses.
    from torch._inductor.runtime.triton_compat import HAS_TRITON, triton_key

    if HAS_TRITON:
        triton_key()


def caching_device_properties():
    for _, device_interface in get_registered_device_interfaces():
        if device_interface.is_available():
            device_interface.Worker.get_device_properties()


def _compile_start() -> None:
    global _t0, _triton_kernel_metrics
    if _t0 is None:
        _t0 = time()
    if _triton_kernel_metrics is None:
        _triton_kernel_metrics = {}


def _compile_end() -> None:
    global _cumulative_compile_time, _t0, _triton_kernel_metrics
    if _t0 is not None:
        t1 = time()
        _cumulative_compile_time += t1 - _t0
        _t0 = None
        # print("CUMULATIVE COMPILE TIME", _cumulative_compile_time)
    if _triton_kernel_metrics:
        # Log triton kernel info
        sorted_info = dict(sorted(_triton_kernel_metrics.items()))
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "triton_kernel_info",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(sorted_info),
        )
        _triton_kernel_metrics = None


def _add_triton_kernel_info(kernel_name: str, info: dict[str, Any]):
    global _triton_kernel_metrics
    # Must be called between _compile_start and _compile_end
    if _triton_kernel_metrics is not None:
        _triton_kernel_metrics[kernel_name] = info


def _emit_triton_kernel_compile_metric(
    kernel_name: str,
    elapsed_us: int,
    autotune_cache_info: dict[str, Any] | None = None,
) -> None:
    """Emit per-kernel ``compile_time_us`` to both the dynamo
    ``_triton_kernel_metrics`` map and the ``MetricsContext`` top-N.
    Used by both the standard ``AsyncCompile.triton`` path and the
    ``PipelineCachingAutotunerPlugin``."""
    info = dict(autotune_cache_info) if autotune_cache_info else {}
    info["compile_time_us"] = elapsed_us
    _add_triton_kernel_info(kernel_name, info)
    get_metrics_context().add_top_n(
        "triton_kernel_compile_times_us", kernel_name, elapsed_us
    )


_IS_WINDOWS = sys.platform == "win32"

log = logging.getLogger(__name__)

# Used to keep track of all process pools invoked so far.
_pool_set = OrderedSet[AnyPool]()


def shutdown_compile_workers() -> None:
    """Shut down all outstanding compile-worker pools."""
    for pool in _pool_set:
        pool.shutdown()
    AsyncCompile._ready_future = None
    after_fork()


def after_fork():
    """Reset pools to initial state without shutting them down"""
    _pool_set.clear()
    AsyncCompile.process_pool.cache_clear()


try:
    os.register_at_fork(after_in_child=after_fork)
except AttributeError:
    pass  # register_at_fork does not exist on windows


def get_compile_threads() -> int:
    """
    Temporary for internal rollout. Assign config.compile_threads lazily and return it.
    TODO: remove after rollout.
    """
    if config.compile_threads is None:
        config.compile_threads = config.decide_compile_threads()
    return config.compile_threads


@clear_on_fresh_cache
class CompiledTritonKernels:
    """
    In memory cache for storing compiled triton kernels.

    Each triton kernel is keyed by the hash of its source code. Each value stored
    in the cache is a return value of AsyncCompile.triton().

    Currently, the cache stores Future objects, but it should be generalizable for any kernels.
    """

    _cache: dict[str, CodeCacheFuture] = {}

    @staticmethod
    def key(kernel_src: str):
        """
        Generates a cache key given a triton kernel's full source code.
        This source includes the inductor meta, compilation metadata, the kernel itself, etc.
        `kernel_src` should be the exact string passed to async_compile.triton()'s first argument.
        """
        # Hashes the kernel source with torch_key into a single hash key
        return code_hash(kernel_src, extra=torch_key())

    @staticmethod
    def save(kernel_src: str, future: CodeCacheFuture):
        """
        Saves a compiled triton kernel to the cache.
        TODO: We store a LambdaFuture as that's the callable returned by async_compile.triton,
        but the real type we want to return here is actually an abstract triton kernel.

        TODO: Source code here is not just the kernel's source code, but also includes the inductor preamble, etc.
        so it could be less strict.
        """
        key = CompiledTritonKernels.key(kernel_src)
        CompiledTritonKernels._cache[key] = future

    @staticmethod
    def get(kernel_src: str) -> CodeCacheFuture | None:
        key = CompiledTritonKernels.key(kernel_src)
        return CompiledTritonKernels._cache.get(key, None)

    @staticmethod
    def cache_clear():
        CompiledTritonKernels._cache = {}

    @staticmethod
    def remove_future(kernel_src: str) -> None:
        key = CompiledTritonKernels.key(kernel_src)

        # Delete the LambdaFuture if there is one
        if key in CompiledTritonKernels._cache:
            del CompiledTritonKernels._cache[key]


class AsyncCompile:
    """
    Utilities to compile in thread pools or subprocess pools (in the case of Triton).
    """

    _ready_future: Future[Any] | None = None
    _metal_sources: list[tuple[str, str, list[str]]] | None = None

    def __init__(self) -> None:
        pass

    @staticmethod
    @functools.lru_cache(1)
    def pool() -> ThreadPoolExecutor:
        assert get_compile_threads() > 1
        return ThreadPoolExecutor(get_compile_threads())

    @staticmethod
    def _get_ready():
        """No-op function to help mark when the subprocess pool is ready."""
        return "ready"

    @staticmethod
    @functools.lru_cache(1)
    def process_pool() -> AnyPool:
        assert get_compile_threads() > 1
        AsyncCompile._ready_future = None
        log.info(
            "Creating '%s' pool with %d workers",
            config.worker_start_method,
            get_compile_threads(),
        )

        pool: AnyPool
        if config.worker_start_method == "subprocess":
            # Wrapper around ProcessPoolExecutor forks in a new process we control
            pool = SubprocPool(
                get_compile_threads(), quiesce=config.quiesce_async_compile_pool
            )
        else:
            if config.worker_start_method == "spawn":
                # Avoid creating pools in the spawned subprocs themselves:
                os.environ["TORCH_WARM_POOL"] = "0"
            pre_fork_setup()
            ctx = multiprocessing.get_context(config.worker_start_method)
            pool = TrackedProcessPoolExecutor(
                get_compile_threads(),
                mp_context=ctx,
                initializer=partial(_async_compile_initializer, os.getpid()),
            )
            # when this pool is created in a subprocess object, the normal exit handler
            # doesn't run, and we need to register our own handler.
            # exitpriority has to be high, because another one of the finalizers will
            # kill the worker thread that sends the shutdown message to the workers...
            multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)

        _pool_set.add(pool)
        return pool

    @classmethod
    def warm_pool(cls) -> None:
        if get_compile_threads() <= 1:
            return
        _compile_start()
        # Pool is created on first access. Note for a SubprocPool, the sidecar process starts,
        # but its ProcessPoolExecutor does not initialize until a wakeup() call or the first
        # job is submitted.
        cls.process_pool()
        _compile_end()

    @classmethod
    def wait_pool_ready(cls, timeout=120) -> None:
        cls.use_process_pool()
        if cls._ready_future is not None:
            cls._ready_future.result(timeout=timeout)

    @classmethod
    def submit(cls, task: Callable[..., Any]) -> Any:
        if get_compile_threads() <= 1:
            return task()
        return cls.pool().submit(task)

    @classmethod
    def use_process_pool(cls):
        if get_compile_threads() <= 1:
            return False

        # Proton instrumentation backend requires compilation to happen in the main
        # process so it can instrument the Triton IR during JIT compilation.
        # Force synchronous compilation when proton profiling is enabled.
        if config.triton.proton_profiling:
            return False

        # Create a dummy job to check if the pool is ready. Submit it here instead of at
        # pool creation so we don't launch the full pool of worker subprocesses until
        # we're sure they're needed.
        if not cls._ready_future:
            cls._ready_future = cls.process_pool().submit(cls._get_ready)
        return cls._ready_future.done()

    @classmethod
    def wakeup(cls) -> None:
        """
        If using a SubprocPool, signal the sidecar process to start up its
        ProcessPoolExecutor.
        """
        if not cls.use_process_pool():
            return
        pool = cls.process_pool()
        if isinstance(pool, SubprocPool):
            pool.wakeup()

    def triton(self, kernel_name: str, source_code: str, device_str: str = "cuda"):
        """
        Async_compile.triton is more complicated than the other backends because
        we're trying to optimize compile time as much as possible for this hot callsite.

        First of all, the function is cached by CompiledTritonKernels; if there's a kernel
        already compiled, we grab it directly from the cache and return.

        Otherwise, if we have multiple compile threads, we kick off triton compilations on each
        worker process by giving it a kernel and source code to compile. The worker initializes
        a CachingAutotuner, runs triton compilation, and pickles the kernel back to us.
        We use TritonCompileResult to represent the objects being pickled back to us by each
        worker.

        Some maybe not obvious things that are pickled back to us:
        - Most of the time, we can avoid sending back CachingAutotuner.fn and other metadata
          and do not have to pay the cost of loading the triton kernel on the parent. But certain
          cases, like coordesc tuning and dynamic_scale_rblock, require us to reload the function
          in the parent lazily when we require it.
        - The AutotuneCache, if enabled, is constructed on each worker per triton config
          and pickled by to us via `CachingAutotuner.save_cache_hook`.
        """
        load_kernel = functools.partial(
            _load_triton_kernel_from_source, kernel_name, source_code
        )

        def reload_kernel_in_parent():
            # Benchmark how often this happens
            with dynamo_timed("reload_kernel_in_parent"):
                return load_kernel()

        counters["inductor"]["async_compile_cache_miss"] += 1

        kernel_code_log.info("Triton Kernel:\n%s", source_code)
        _compile_start()

        if os.environ.get("TRITON_INTERPRET", "0") == "1":
            return getattr(
                torch._inductor.codecache.PyCodeCache.load(source_code), kernel_name
            )

        is_parallel = self.use_process_pool()
        set_feature_use("parallel_compile_post_warmup", is_parallel)

        compile_id = torch._guards.CompileContext.current_compile_id()
        is_backward = getattr(V.graph, "is_backward", False)

        if (future := CompiledTritonKernels.get(source_code)) is not None:
            counters["inductor"]["async_compile_cache_hit"] += 1
            # Set reload_kernel_from_src properly based on source_code
            if isinstance(future, StaticAutotunerFuture):
                # Remove the future now that we've cache hit
                CompiledTritonKernels.remove_future(source_code)
                future.reload_kernel_from_src = reload_kernel_in_parent
            if is_parallel:
                return future
            else:
                return future.result()

        # Cache miss
        if is_parallel:
            # Ensure libdevice path is set in os.environ before passing to workers
            _set_triton_libdevice_path()
            # We want to support changing these env vars after (and while) the
            # process pool is running, so pass them to the subprocess to reset.
            env_vars = [
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "TRITON_LIBDEVICE_PATH",
            ]
            extra_env = {v: os.environ[v] for v in env_vars if v in os.environ}
            extra_config = {
                "use_static_triton_launcher": torch._inductor.config.use_static_triton_launcher
            }

            if len(torch._inductor.config.autotune_lookup_table) > 0:
                m = size_hints_regex.search(source_code)
                if m:
                    size_hints_str = m.group(1)
                else:
                    size_hints_str = str(None)

                triton_src = source_code.split("@triton.jit\n")[1]
                from torch._inductor.runtime.triton_heuristics import (
                    generate_lookup_hash_from_source_code,
                )

                fn_hash = generate_lookup_hash_from_source_code(
                    size_hints_str, triton_src
                )

                if fn_hash in torch._inductor.config.autotune_lookup_table:
                    extra_config["autotune_lookup_table"] = {  # type: ignore[assignment]
                        fn_hash: torch._inductor.config.autotune_lookup_table[fn_hash]
                    }

            if config.pipeline_caching_autotuner:
                sock_path, authkey, listen_sock = _setup_streaming_listener(
                    kernel_name
                )
            else:
                sock_path = None
                authkey = None
                listen_sock = None

            task = self.process_pool().submit(
                _worker_compile_triton,
                load_kernel,
                extra_env,
                extra_config,
                streaming_address=sock_path,
                streaming_authkey=authkey,
            )

            if config.pipeline_caching_autotuner:
                # If the worker dies before connecting, accept() blocks
                # forever. Closing the listener unblocks it with OSError.
                assert listen_sock is not None

                def _on_task_fail_close_listener(future: Future) -> None:
                    if future.exception() is not None:
                        try:
                            listen_sock.close()
                        except OSError:
                            pass

                task.add_done_callback(_on_task_fail_close_listener)

            def get_result() -> CachingAutotuner:
                elapsed_us: int | None = None
                if config.pipeline_caching_autotuner:
                    assert listen_sock is not None and sock_path is not None
                    assert authkey is not None
                    try:
                        # Times out per listen_sock.settimeout in setup.
                        conn_sock, _ = listen_sock.accept()
                    except OSError:
                        # Re-raises if the worker failed; otherwise spurious
                        # close or timeout - surface as a generic stream error.
                        try:
                            task.result()
                        except SubprocException as e:
                            raise e.with_name(kernel_name) from e
                        raise RuntimeError(
                            f"streaming listener closed before {kernel_name} connected"
                        )
                    finally:
                        try:
                            os.unlink(sock_path)
                        except OSError:
                            pass
                    listen_sock.close()
                    conn = Connection(conn_sock.detach())
                    # mp.connection challenge/answer handshake: rejects any
                    # peer that doesn't share authkey.
                    deliver_challenge(conn, authkey)
                    answer_challenge(conn, authkey)
                    if not conn.poll(_STREAMING_TIMEOUT_S):
                        raise TimeoutError(
                            f"streaming worker for {kernel_name} sent no kernel "
                            f"for {_STREAMING_TIMEOUT_S:.0f}s"
                        )
                    try:
                        kernel_bytes = conn.recv_bytes()
                    except EOFError:
                        try:
                            task.result()
                        except SubprocException as e:
                            raise e.with_name(kernel_name) from e
                        raise RuntimeError(
                            f"worker for {kernel_name} closed pipe before sending kernel"
                        )
                    kernel = cast(
                        "CachingAutotuner",
                        _StreamingUnpickler(io.BytesIO(kernel_bytes)).load(),
                    )
                else:
                    try:
                        result = task.result()
                    except SubprocException as e:
                        raise e.with_name(kernel_name) from e
                    kernel_or_none, elapsed_us = result
                    assert kernel_or_none is not None, (
                        "non-streaming worker must return a kernel"
                    )
                    kernel = kernel_or_none

                # Now that we've compiled, we should clear the future
                # so it can't be used again
                kernel.set_compile_info(compile_id, is_backward)
                CompiledTritonKernels.remove_future(source_code)

                kernel.restore_after_unpickle(old_values=None)

                if config.pipeline_caching_autotuner:
                    # Stash the handle + ``_reload_kernel`` (normally set by
                    # ``precompile``) so coordesc / rblock scaling can still
                    # reload the JITFunction. The bg drain thread below does
                    # the precompile work.
                    handle = PipelineCachingAutotunerHandle(
                        launcher_q=_queue.Queue(),
                        drain_future=Future(),
                        num_configs=len(kernel.configs or []),
                        kernel_name=kernel_name,
                    )
                    kernel._pipeline_caching_autotuner_handle = handle
                    kernel._reload_kernel = reload_kernel_in_parent

                    threading.Thread(
                        target=_bg_drain_kernel,
                        args=(
                            kernel,
                            conn,
                            handle,
                            CompiledTritonKernels.key(source_code),
                        ),
                        name=f"inductor-stream-bg-drain-{kernel_name}",
                        daemon=True,
                    ).start()
                else:
                    kernel.precompile(
                        warm_cache_only=False,
                        reload_kernel=reload_kernel_in_parent,
                        static_triton_bundle_key=CompiledTritonKernels.key(
                            source_code
                        ),
                    )
                    assert elapsed_us is not None
                    _emit_triton_kernel_compile_metric(
                        kernel_name,
                        elapsed_us,
                        autotune_cache_info=kernel.autotune_cache_info,
                    )
                return kernel

            future = LambdaFuture(get_result, future=task)
            CompiledTritonKernels.save(source_code, future)
            return future
        else:
            with dynamo_timed(
                "async_compile.precompile",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="triton_compile_time_us",
                log_waitcounter=True,
                waitcounter_name_override="compile_triton",
            ):
                fail = None
                try:
                    start_ns = time_ns()
                    _set_triton_ptxas_path()
                    _set_triton_libdevice_path()
                    kernel = load_kernel()
                    kernel.set_compile_info(compile_id, is_backward)
                    kernel.precompile(
                        warm_cache_only=False,
                        static_triton_bundle_key=CompiledTritonKernels.key(source_code),
                    )
                    elapsed_us = (time_ns() - start_ns) // 1000
                    _emit_triton_kernel_compile_metric(
                        kernel_name,
                        elapsed_us,
                        autotune_cache_info=kernel.autotune_cache_info,
                    )
                    return kernel
                except Exception as e:
                    fail = str(e)
                    raise
                finally:
                    log_triton_builds(fail=fail)

    def multi_kernel(self, *args, **kwargs) -> Any:
        from torch._inductor.codegen.multi_kernel import MultiKernelCall

        # no need to call this in parallel since the sub-kernels are already parallel tasks
        return MultiKernelCall(*args, **kwargs)

    def size_hint_multi_kernel(self, *args, **kwargs) -> Any:
        from torch._inductor.codegen.multi_kernel import SizeHintMultiKernelCall

        return SizeHintMultiKernelCall(*args, **kwargs)

    def cpp(self, source_code: str):
        kernel_code_log.info("CPP Kernel:\n%s", source_code)
        if get_compile_threads() <= 1:
            return CppCodeCache.load(source_code).kernel
        else:
            get_result = CppCodeCache.load_async(source_code, submit_fn=self.submit)
            return LambdaFuture(lambda: get_result().kernel)

    def cpp_pybinding(self, argtypes: list[str], source_code: str):
        kernel_code_log.info("CPP+Bindings Kernel:\n%s", source_code)
        if get_compile_threads() <= 1:
            return CppPythonBindingsCodeCache.load_pybinding(argtypes, source_code)
        else:
            get_result = CppPythonBindingsCodeCache.load_pybinding_async(
                argtypes, source_code, submit_fn=self.submit
            )
            return LambdaFuture(get_result)

    def cutlass(self, cache_cls, source_code, dst_file_ext, aot_compile=False):
        def task():
            if aot_compile:
                # We rely on JITInductor to compile the CUDA code,
                # so that we can load it into AOTInductor.
                output_path, *_ = cache_cls.compile(source_code, "o")
                cache_cls.aot_kernels_o.append(output_path)
            return cache_cls.load(source_code, dst_file_ext)[0]

        return self.submit(task)

    def cuda(self, source_code, dst_file_ext, aot_compile=False):
        kernel_code_log.info("CUDA Kernel:\n%s", source_code)
        return self.cutlass(CUDACodeCache, source_code, dst_file_ext, aot_compile)

    def xpu(self, source_code, dst_file_ext, aot_compile=False):
        kernel_code_log.info("XPU Kernel:\n%s", source_code)
        return self.cutlass(XPUCodeCache, source_code, dst_file_ext, aot_compile)

    def rocm(
        self,
        source_code,
        dst_file_ext,
        aot_compile=False,
    ):
        kernel_code_log.info("ROCm Kernel:\n%s", source_code)

        def task():
            if aot_compile:
                output_path, *_ = ROCmCodeCache.compile(source_code, dst_file_ext="o")
                ROCmCodeCache.aot_kernels_o.append(output_path)
            if config.rocm.generate_test_runner:
                _ = ROCmCodeCache.compile(source_code, dst_file_ext="exe")
            return ROCmCodeCache.load(source_code, dst_file_ext)[0]

        return self.submit(task)

    def halide(self, meta: HalideMeta, source_code: str):
        kernel_code_log.info("Halide Kernel:\n%r\n%s", meta, source_code)
        if get_compile_threads() <= 1:
            return HalideCodeCache.generate_halide(meta, source_code)
        else:
            get_result = HalideCodeCache.generate_halide_async(
                meta, source_code, submit_fn=self.submit
            )
            return LambdaFuture(get_result)

    def cutedsl(self, kernel_name: str, source_code: str):
        """
        Compile CuteDSL (CUTLASS Python DSL) kernels.

        Args:
            kernel_name: Name of the kernel to be defined
            source_code: Source code of the CuteDSL kernel, as a string

        Note:
            CuteDSL currently requires source files to do its compilation, there we
            use the PyCodeCache to write the source code to a file and load it.
        """
        from torch._inductor.codegen.cutedsl.cutedsl_kernel import (
            CuteDSLKernelWrapper,
            MAIN_SUFFIX,
        )

        kernel_code_log.info("CuteDSL Kernel:\n%s", source_code)

        def task():
            key, path = torch._inductor.codecache.PyCodeCache.write(source_code)
            mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)

            # Find our special entry point named function
            main_func_name = f"{kernel_name}_{MAIN_SUFFIX}"
            if not hasattr(mod, main_func_name):
                available = [name for name in dir(mod) if callable(getattr(mod, name))]
                raise RuntimeError(
                    f"Could not find CuteDSL main kernel function '{main_func_name}'. Available callables: {available}"
                )

            return CuteDSLKernelWrapper(getattr(mod, main_func_name), kernel_path=path)

        if get_compile_threads() <= 1:
            return task()
        else:
            future = self.submit(task)
            return LambdaFuture(lambda: future.result())

    def pallas(self, kernel_name: str, source_code: str):
        """
        Compile Pallas (JAX experimental) kernels.

        Args:
            kernel_name: Name of the kernel to be defined
            source_code: Source code of the Pallas kernel, as a string

        Note:
            Pallas kernels are Python code that uses JAX and Pallas APIs.
            We use the PyCodeCache to write the source code to a file and load it.
        """
        from torch._inductor.codegen.pallas import MAIN_SUFFIX, PallasKernelWrapper

        kernel_code_log.info("Pallas Kernel:\n%s", source_code)

        def task():
            key, path = torch._inductor.codecache.PyCodeCache.write(source_code)
            mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)

            # Find our special entry point named function
            main_func_name = f"{kernel_name}_{MAIN_SUFFIX}"
            if not hasattr(mod, main_func_name):
                available = [name for name in dir(mod) if callable(getattr(mod, name))]
                raise RuntimeError(
                    f"Could not find Pallas main kernel function '{main_func_name}'. Available callables: {available}"
                )

            return PallasKernelWrapper(getattr(mod, main_func_name), kernel_path=path)

        if get_compile_threads() <= 1:
            return task()
        else:
            future = self.submit(task)
            return LambdaFuture(lambda: future.result())

    def nv_universal_gemm(self, kernel_name: str, source_code: str):
        """
        Compile NVIDIA Universal GEMM kernels.

        Args:
            kernel_name: Name of the kernel to be defined
            source_code: Source code of the kernel, as a string

        Note:
            NVIDIA Universal GEMM kernels are Python code that calls the cutlass_api library.
            We use the PyCodeCache to write the source code to a file and load it.
        """
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            NVUniversalGemmKernelWrapper,
        )
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_scheduling import (
            MAIN_SUFFIX,
        )

        kernel_code_log.info("NVIDIA Universal GEMM Kernel:\n%s", source_code)

        def task():
            key, path = torch._inductor.codecache.PyCodeCache.write(source_code)
            mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)

            # Find our special entry point named function
            main_func_name = f"{kernel_name}_{MAIN_SUFFIX}"
            if not hasattr(mod, main_func_name):
                available = [name for name in dir(mod) if callable(getattr(mod, name))]
                raise RuntimeError(
                    f"Could not find NVIDIA Universal GEMM main kernel function "
                    f"'{main_func_name}'. Available callables: {available}"
                )

            return NVUniversalGemmKernelWrapper(
                getattr(mod, main_func_name), kernel_path=path
            )

        if get_compile_threads() <= 1:
            return task()
        else:
            future = self.submit(task)
            return LambdaFuture(lambda: future.result())

    def metal(self, kernel_name: str, source: str, headers: list[str]) -> None:
        """Register a Metal kernel body; wait() compiles all registered kernels into one library."""
        if self._metal_sources is None:
            self._metal_sources = []
        self._metal_sources.append((kernel_name, source, headers))

    def wait(self, scope: dict[str, Any]) -> None:
        if get_compile_threads() > 1:
            with dynamo_timed(
                "async_compile.wait",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="triton_compile_time_us",
                log_waitcounter=True,
                waitcounter_name_override="compile_triton",
            ):
                self._wait_futures(scope)

        if self._metal_sources:
            from torch._inductor.runtime.runtime_utils import compile_mps_shaders

            scope.update(compile_mps_shaders(self._metal_sources))
            self._metal_sources.clear()

        _compile_end()

    def _wait_futures(self, scope: dict[str, Any]) -> None:
        kernels = {
            key: value
            for key, value in scope.items()
            if isinstance(value, (Future, CodeCacheFuture))
        }
        pbar = tqdm(
            total=len(kernels),
            desc="Inductor Compilation",
            disable=config.disable_progress,
            delay=0,
        )
        # compile_worker_wait_timeout=0 (default) means "wait forever"; map
        # it to None so both Future.result() and CodeCacheFuture.result()
        # receive the same "no timeout" sentinel.
        wait_timeout = config.compile_worker_wait_timeout or None
        for key, result in kernels.items():
            if config.verbose_progress and not isinstance(pbar, _Faketqdm):
                pbar.set_postfix_str(key)
            try:
                kernel = result.result(timeout=wait_timeout)
                scope[key] = kernel
            except FuturesTimeoutError as e:
                # concurrent.futures.TimeoutError became an alias of the
                # builtin TimeoutError in Python 3.11; on 3.10 it is a
                # distinct class, so catch it explicitly.
                raise RuntimeError(
                    f"Inductor compile-worker future for {key!r} did not "
                    f"complete within {wait_timeout}s. Override with "
                    "TORCHINDUCTOR_COMPILE_WORKER_WAIT_TIMEOUT=<seconds>."
                ) from e
            except BrokenProcessPool as e:
                raise RuntimeError(
                    "A compilation subprocess exited unexpectedly. This "
                    "is likely due to a crash. To facilitate debugging, "
                    "you can re-run with TORCHINDUCTOR_COMPILE_THREADS=1 "
                    "to cause compilation to occur in the main process."
                ) from e
            pbar.update(1)


def maybe_warm_pool() -> None:
    if (
        os.environ.get("TORCH_TNT_IN_USE", "0") == "1"
        or os.environ.get("TORCH_WARM_POOL", "1") != "1"
        # The subprocess pool is only used for the Triton backend
        or not has_triton_package()
        # Skip for fbcode. We have internal reports of usages inside multiprocessing
        # pools that lead a multiplicative number of compile subprocesses.
        or config.is_fbcode()
    ):
        return

    AsyncCompile.warm_pool()
    # TODO: This starts the SubprocPool's internal process pool as early as possible at
    # the expense of creating a bunch of worker processes that might not be needed. We
    # could start them lazily if we're willing to lose a small amount of compile time.
    AsyncCompile.wakeup()


# On exit give the workers a chance to clean themselves up. Without this the
# resource_tracker can complain about leaked semaphores coming from the
# ProcessPoolExecutor:
#   UserWarning: resource_tracker: There appear to be 5 leaked semaphore objects
#   to clean up at shutdown
atexit.register(shutdown_compile_workers)
