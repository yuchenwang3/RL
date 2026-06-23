# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP helpers for sparse vLLM refit payload transfer."""

from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import suppress
from typing import Any, cast

import torch

from nemo_rl.distributed.virtual_cluster import _get_free_port_local, _get_node_ip_local
from nemo_rl.utils.weight_transfer_delta_tracker import DeltaCompressionTracker
from nemo_rl.utils.weight_transfer_protocol import (
    G_DENSE_TRANSPORT,
    G_SPARSE_INDICES_TRANSPORT,
    NamedTensor,
    TensorBatch,
    TensorPayload,
    env_int,
    get_target_packed_tensor_size,
    is_refit_receiver_timing,
    next_chunk,
)

G_VLLM_REFIT_SPARSE_DELTA_PATH = "/nemo-rl/refit/sparse-delta"
G_VLLM_REFIT_HEALTH_PATH = "/nemo-rl/refit/health"
G_VLLM_REFIT_FLUSH_PATH = "/nemo-rl/refit/flush"
G_VLLM_GENERATE_PATH = "/nemo-rl/generate"
G_VLLM_REFIT_API_KEY_HEADER = "x-nemo-rl-refit-key"
G_VLLM_REFIT_UNCOMPRESSED_BYTES_HEADER = "x-nemo-rl-refit-uncompressed-bytes"
G_VLLM_REFIT_ASYNC_RECEIVER_APPLY_ENV = "NRL_REFIT_ASYNC_RECEIVER_APPLY"
G_VLLM_REFIT_HTTP_POOL_MAXSIZE_ENV = "NRL_REFIT_HTTP_POOL_MAXSIZE"
G_VLLM_REFIT_HTTP_POST_PARALLELISM_ENV = "NRL_REFIT_HTTP_POST_PARALLELISM"
G_VLLM_REFIT_HTTP_INFLIGHT_BUCKETS_ENV = "NRL_REFIT_HTTP_INFLIGHT_BUCKETS"
G_VLLM_REFIT_HTTP_FANOUT_WORKERS_ENV = "NRL_REFIT_HTTP_FANOUT_WORKERS"
G_VLLM_REFIT_HTTP_BODY_COMPRESS_ENV = "NRL_REFIT_HTTP_BODY_COMPRESS"
G_VLLM_REFIT_HTTP_PROGRESS_INTERVAL_ENV = "NRL_REFIT_HTTP_PROGRESS_INTERVAL_S"
G_VLLM_REFIT_HTTP_EXPORT_CHUNK_BYTES_ENV = "NRL_REFIT_HTTP_EXPORT_CHUNK_BYTES"
G_VLLM_REFIT_HTTP_ENCODE_WORKERS_ENV = "NRL_REFIT_HTTP_ENCODE_WORKERS"
G_VLLM_REFIT_HTTP_ZSTD_THREADS_ENV = "NRL_REFIT_HTTP_ZSTD_THREADS"
G_GENERATION_RESPONSE_KEYS = (
    "output_ids",
    "logprobs",
    "generation_lengths",
    "unpadded_sequence_lengths",
    "truncated",
)
G_DEFAULT_INFLIGHT_BUCKETS = 1
G_DEFAULT_PROGRESS_INTERVAL_S = 30
G_DEFAULT_HTTP_EXPORT_CHUNK_BYTES = 512 * 1024**2

_HTTP_SESSION_LOCAL = threading.local()
_EXECUTORS: dict[tuple[str, int], ThreadPoolExecutor] = {}
_EXECUTORS_LOCK = threading.Lock()


def _get_keepalive_session() -> Any:
    session = getattr(_HTTP_SESSION_LOCAL, "session", None)
    if session is not None:
        return session
    try:
        import requests
    except ImportError:
        _HTTP_SESSION_LOCAL.session = False
        return False
    pool_size = env_int(G_VLLM_REFIT_HTTP_POOL_MAXSIZE_ENV, default=64, min_value=1)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    _HTTP_SESSION_LOCAL.session = session
    return session


def normalize_vllm_refit_base_urls(refit_urls: Sequence[str]) -> list[str]:
    suffixes = (
        G_VLLM_REFIT_SPARSE_DELTA_PATH,
        G_VLLM_REFIT_HEALTH_PATH,
        G_VLLM_REFIT_FLUSH_PATH,
        G_VLLM_GENERATE_PATH,
    )
    urls = []
    for raw_url in refit_urls:
        url = raw_url.strip()
        for suffix in suffixes:
            if url.endswith(suffix):
                url = url[: -len(suffix)]
                break
        if url:
            urls.append(url.rstrip("/"))
    return urls


def vllm_refit_sparse_delta_url(base_url: str) -> str:
    return base_url.rstrip("/") + G_VLLM_REFIT_SPARSE_DELTA_PATH


def vllm_refit_health_url(base_url: str) -> str:
    return base_url.rstrip("/") + G_VLLM_REFIT_HEALTH_PATH


def vllm_refit_flush_url(base_url: str) -> str:
    return base_url.rstrip("/") + G_VLLM_REFIT_FLUSH_PATH


def vllm_generate_url(base_url: str) -> str:
    return base_url.rstrip("/") + G_VLLM_GENERATE_PATH


def vllm_refit_api_key_headers(api_key_env_var: str | None) -> dict[str, str]:
    if not api_key_env_var:
        return {}
    expected = os.environ.get(api_key_env_var)
    if not expected:
        raise RuntimeError(
            "vLLM HTTP refit API key env var "
            f"{api_key_env_var!r} is configured but unset or empty."
        )
    return {G_VLLM_REFIT_API_KEY_HEADER: expected}


def vllm_refit_api_key_is_valid(
    api_key_env_var: str | None,
    headers: Mapping[str, str],
) -> bool:
    if not api_key_env_var:
        return True
    expected = os.environ.get(api_key_env_var)
    return bool(expected) and headers.get(G_VLLM_REFIT_API_KEY_HEADER) == expected


def _http_post_parallelism(url_count: int) -> int:
    value = env_int(
        G_VLLM_REFIT_HTTP_POST_PARALLELISM_ENV,
        default=max(1, url_count),
        min_value=1,
    )
    return min(value, max(1, url_count))


def _http_fanout_parallelism(url_count: int) -> int:
    return env_int(
        G_VLLM_REFIT_HTTP_FANOUT_WORKERS_ENV,
        default=_http_post_parallelism(url_count),
        min_value=1,
    )


def _inflight_bucket_window() -> int:
    return env_int(
        G_VLLM_REFIT_HTTP_INFLIGHT_BUCKETS_ENV,
        default=G_DEFAULT_INFLIGHT_BUCKETS,
        min_value=1,
    )


def _http_export_chunk_size() -> int:
    requested = env_int(
        G_VLLM_REFIT_HTTP_EXPORT_CHUNK_BYTES_ENV,
        default=G_DEFAULT_HTTP_EXPORT_CHUNK_BYTES,
        min_value=1,
    )
    if not torch.cuda.is_available():
        return requested
    return min(requested, get_target_packed_tensor_size())


def _http_encode_workers() -> int:
    return env_int(
        G_VLLM_REFIT_HTTP_ENCODE_WORKERS_ENV,
        default=1,
        min_value=1,
    )


def _http_body_compression_mode() -> str | None:
    raw_mode = os.getenv(G_VLLM_REFIT_HTTP_BODY_COMPRESS_ENV, "")
    mode = raw_mode.strip().lower()
    if mode in {"", "0", "false", "none", "off"}:
        return None
    if mode in {"1", "true", "yes", "on"}:
        return "zstd"
    if mode not in {"zlib", "zstd"}:
        raise ValueError(
            f"Unsupported vLLM HTTP refit body compression {raw_mode!r}; "
            "expected zlib or zstd."
        )
    return mode


def _format_http_timing_value(key: str, value: float | int) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    precision = 1 if key in {"posted_mb", "export_chunk_mb"} else 3
    return f"{float(value):.{precision}f}"


def encode_vllm_refit_request_body(body: bytes) -> tuple[bytes, dict[str, str]]:
    return _encode_vllm_refit_request_body(body, _http_body_compression_mode())


def _encode_vllm_refit_request_body(
    body: bytes,
    mode: str | None,
) -> tuple[bytes, dict[str, str]]:
    if mode is None or not body:
        return body, {}
    if mode == "zlib":
        encoded = zlib.compress(body, level=1)
    elif mode == "zstd":
        encoded = _zstd_compress(body)
    else:
        raise ValueError(f"Unsupported vLLM HTTP refit request encoding {mode!r}.")
    return (
        encoded,
        {
            "content-encoding": mode,
            G_VLLM_REFIT_UNCOMPRESSED_BYTES_HEADER: str(len(body)),
        },
    )


def _refit_request_body_encoding(headers: Mapping[str, str]) -> str | None:
    encoding = headers.get("content-encoding") or headers.get("Content-Encoding")
    if encoding is None or encoding.strip().lower() in {"", "identity"}:
        return None
    mode = encoding.strip().lower()
    if mode in {"zlib", "zstd"}:
        return mode
    raise ValueError(f"Unsupported vLLM HTTP refit request encoding {encoding!r}.")


def decode_vllm_refit_request_body(
    body: bytes,
    headers: Mapping[str, str],
) -> bytes:
    mode = _refit_request_body_encoding(headers)
    if mode is None:
        return body
    if mode == "zlib":
        return zlib.decompress(body)
    if mode == "zstd":
        return _zstd_decompress(body)
    raise AssertionError(f"Validated unsupported request encoding {mode!r}.")


async def decode_vllm_refit_request_body_async(
    body: bytes,
    headers: Mapping[str, str],
) -> bytes:
    if _refit_request_body_encoding(headers) is None:
        return body
    return await asyncio.to_thread(decode_vllm_refit_request_body, body, headers)


def _executor(key: str, workers: int) -> ThreadPoolExecutor:
    with _EXECUTORS_LOCK:
        cache_key = (key, workers)
        if cache_key not in _EXECUTORS:
            _EXECUTORS[cache_key] = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix=f"nrl-{key}",
            )
        return _EXECUTORS[cache_key]


def _map_parallel(
    key: str,
    items: Sequence[Any],
    fn: Callable[[Any], Any],
    parallelism: int,
) -> list[Any]:
    if parallelism == 1 or len(items) == 1:
        return [fn(item) for item in items]
    return list(_executor(key, parallelism).map(fn, items))


def _validate_generation_response(result: dict[str, Any], batch_size: int) -> None:
    if not result.get("ok", False):
        raise RuntimeError(f"vLLM HTTP generation shard failed: {result}")
    missing = [
        key
        for key in G_GENERATION_RESPONSE_KEYS
        if not isinstance(result.get(key), list) or len(result[key]) != batch_size
    ]
    if missing:
        raise RuntimeError(f"Incomplete vLLM HTTP generation response: {missing}")


def post_generation_payload_to_urls(
    generation_urls: Sequence[str],
    payload: dict[str, Any],
    *,
    api_key_env_var: str | None = None,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    urls = normalize_vllm_refit_base_urls(generation_urls)
    input_ids = payload.get("input_ids")
    input_lengths = payload.get("input_lengths")
    if not urls:
        raise ValueError("At least one vLLM generation URL is required")
    if not isinstance(input_ids, list) or not isinstance(input_lengths, list):
        raise TypeError("Generation payload requires list input_ids and input_lengths")
    if len(input_ids) != len(input_lengths):
        raise ValueError("Generation payload input_ids/input_lengths length mismatch")
    batch_size = len(input_ids)
    if batch_size == 0:
        return {"ok": True, **{key: [] for key in G_GENERATION_RESPONSE_KEYS}}
    if len(urls) == 1:
        result = _http_request_json(
            vllm_generate_url(urls[0]),
            json.dumps(payload).encode("utf-8"),
            api_key_env_var=api_key_env_var,
            timeout_s=timeout_s,
            content_type="application/json",
        )
        _validate_generation_response(result, batch_size)
        return {key: result[key] for key in ("ok", *G_GENERATION_RESPONSE_KEYS)}

    shard_count = min(len(urls), batch_size)
    shards = [
        (url, list(range(shard_idx, batch_size, shard_count)))
        for shard_idx, url in enumerate(urls[:shard_count])
    ]

    def post_one(url: str, indices: list[int]) -> tuple[list[int], dict[str, Any]]:
        shard = dict(payload)
        for key in ("input_ids", "input_lengths", "stop_strings"):
            if isinstance(payload.get(key), list):
                shard[key] = [payload[key][idx] for idx in indices]
        return (
            indices,
            _http_request_json(
                vllm_generate_url(url),
                json.dumps(shard).encode("utf-8"),
                api_key_env_var=api_key_env_var,
                timeout_s=timeout_s,
                content_type="application/json",
            ),
        )

    results = _map_parallel(
        "refit-gen",
        shards,
        lambda item: post_one(*item),
        _http_post_parallelism(len(shards)),
    )

    merged: dict[str, list[Any]] = {
        key: [None] * batch_size for key in G_GENERATION_RESPONSE_KEYS
    }
    for indices, result in results:
        _validate_generation_response(result, len(indices))
        for result_idx, original_idx in enumerate(indices):
            for key in merged:
                merged[key][original_idx] = result[key][result_idx]
    return {"ok": True, **merged}


def _serialize_refit_payload(payload: TensorPayload) -> bytes:
    payload_tensors, transport, metadata = payload
    if transport not in {G_DENSE_TRANSPORT, G_SPARSE_INDICES_TRANSPORT}:
        raise ValueError(f"vLLM HTTP refit got transport={transport!r}.")
    if transport == G_DENSE_TRANSPORT and metadata:
        raise ValueError("Dense vLLM HTTP refit payloads cannot carry metadata.")
    request = {
        "transport": transport,
        "metadata": metadata,
        "payload_tensors": [
            (name, tensor.detach().cpu()) for name, tensor in payload_tensors
        ],
    }
    buffer = io.BytesIO()
    torch.save(request, buffer)
    return buffer.getvalue()


def check_vllm_refit_health(
    refit_urls: Sequence[str],
    *,
    api_key_env_var: str | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    urls = [
        vllm_refit_health_url(url) for url in normalize_vllm_refit_base_urls(refit_urls)
    ]
    if not urls:
        raise ValueError("At least one vLLM HTTP refit URL is required.")
    _post_refit_body_to_endpoint_urls(
        urls,
        None,
        api_key_env_var=api_key_env_var,
        timeout_s=timeout_s,
    )
    return {"ok": True, "urls": len(urls)}


def _drain_iterator(iterator: Iterable[NamedTensor]) -> None:
    for _ in iterator:
        pass


def init_sparse_delta_baseline_from_iterator(
    iterator: Iterable[NamedTensor],
    *,
    delta_tracker: DeltaCompressionTracker | None,
    is_payload_source: bool,
) -> None:
    if delta_tracker is None:
        if is_payload_source:
            raise RuntimeError("vLLM HTTP sparse refit requires delta compression.")
        _drain_iterator(iterator)
        return
    if not is_payload_source:
        _drain_iterator(iterator)
        return
    if delta_tracker.full_sync_interval <= 1:
        raise ValueError("vLLM HTTP sparse refit requires full_sync_interval > 1.")

    tensor_iterator = iter(iterator)
    pending_item = None
    chunk_count = 0
    initialized = 0
    export_chunk_size = _http_export_chunk_size()
    start_s = time.perf_counter()
    last_progress_s = start_s
    progress_interval_s = env_int(
        G_VLLM_REFIT_HTTP_PROGRESS_INTERVAL_ENV,
        default=G_DEFAULT_PROGRESS_INTERVAL_S,
        min_value=1,
    )
    print(
        "REFIT_BASELINE_INIT event=start mode=remote_sparse_http",
        flush=True,
    )
    while True:
        chunk, pending_item = next_chunk(
            tensor_iterator,
            export_chunk_size,
            pending_item=pending_item,
        )
        if not chunk:
            break
        chunk_count += 1
        is_delta, prepared = delta_tracker.prepare_sparse_delta_payload(
            chunk,
            target_device=None,
        )
        if not is_delta:
            dense_tensors = cast(TensorBatch, prepared)
            delta_tracker.snapshot_pending_full_sync_baseline(dense_tensors)
            initialized += len(dense_tensors)
        now_s = time.perf_counter()
        if now_s - last_progress_s >= progress_interval_s:
            print(
                "REFIT_BASELINE_INIT "
                f"event=progress chunks={chunk_count} tensors={initialized} "
                f"seconds={now_s - start_s:.3f}",
                flush=True,
            )
            last_progress_s = now_s
    delta_tracker.on_sync_succeeded()
    print(
        "REFIT_BASELINE_INIT "
        f"event=end chunks={chunk_count} tensors={initialized} "
        f"seconds={time.perf_counter() - start_s:.3f}",
        flush=True,
    )


def stream_sparse_delta_payloads_via_http(
    iterator: Iterable[NamedTensor],
    *,
    delta_tracker: DeltaCompressionTracker | None,
    is_payload_source: bool,
    refit_urls: Sequence[str],
    api_key_env_var: str | None = None,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    urls = normalize_vllm_refit_base_urls(refit_urls)
    if not urls:
        raise ValueError("At least one vLLM HTTP refit URL is required.")
    if delta_tracker is None:
        if is_payload_source:
            raise RuntimeError("vLLM HTTP sparse refit requires delta compression.")
        _drain_iterator(iterator)
        return {"ok": True, "payloads": 0}
    if not is_payload_source:
        _drain_iterator(iterator)
        return {"ok": True, "payloads": 0}
    if delta_tracker.full_sync_interval <= 1:
        raise ValueError("vLLM HTTP sparse refit requires full_sync_interval > 1.")

    endpoint_urls = [vllm_refit_sparse_delta_url(url) for url in urls]
    window = _inflight_bucket_window()
    executor = _executor("refit-post", window)
    inflight: deque[Any] = deque()
    encode_inflight: deque[Any] = deque()
    tensor_iterator = iter(iterator)
    pending_item = None
    payload_count = 0
    has_dense_payload = False
    posted_bytes = 0
    receiver_timing: dict[str, float] = {}
    export_pull_s = 0.0
    encode_s = encode_wait_s = 0.0
    d2h_s = post_wait_s = post_busy_s = flush_wait_s = 0.0
    serialize_s = compress_s = http_post_s = 0.0
    stream_start = time.perf_counter()
    last_progress_s = stream_start
    progress_interval_s = env_int(
        G_VLLM_REFIT_HTTP_PROGRESS_INTERVAL_ENV,
        default=G_DEFAULT_PROGRESS_INTERVAL_S,
        min_value=1,
    )
    chunk_count = 0
    export_chunk_size = _http_export_chunk_size()
    body_compression_mode = _http_body_compression_mode()
    requested_encode_workers = _http_encode_workers()
    async_encode = (
        requested_encode_workers > 1
        and delta_tracker.is_delta_sync()
        and not delta_tracker.has_pending_full_sync_baseline()
    )
    encode_workers = requested_encode_workers if async_encode else 1
    encode_executor = (
        _executor("refit-encode", encode_workers) if async_encode else None
    )

    def post_payload(
        payload: TensorPayload,
    ) -> tuple[int, float, float, float, float, dict[str, Any]]:
        started = time.perf_counter()
        body = _serialize_refit_payload(payload)
        serialize_elapsed = time.perf_counter() - started
        compress_started = time.perf_counter()
        body, extra_headers = _encode_vllm_refit_request_body(
            body,
            body_compression_mode,
        )
        compress_elapsed = time.perf_counter() - compress_started
        post_started = time.perf_counter()
        responses = _post_refit_body_to_endpoint_urls(
            endpoint_urls,
            body,
            api_key_env_var=api_key_env_var,
            timeout_s=timeout_s,
            extra_headers=extra_headers,
        )
        http_post_elapsed = time.perf_counter() - post_started
        timing = _aggregate_fanout_refit_responses(responses)
        return (
            len(body),
            time.perf_counter() - started,
            serialize_elapsed,
            compress_elapsed,
            http_post_elapsed,
            timing,
        )

    def pop_done_future(futures: deque[Any]) -> Any | None:
        for future in futures:
            if future.done():
                futures.remove(future)
                return future
        return None

    def wait_for_next_future(futures: deque[Any]) -> Any:
        while True:
            future = pop_done_future(futures)
            if future is not None:
                return future
            completed_set, _ = wait(
                list(futures),
                timeout=max(
                    0.1,
                    progress_interval_s - (time.perf_counter() - last_progress_s),
                ),
                return_when=FIRST_COMPLETED,
            )
            if completed_set:
                future = next(iter(completed_set))
                futures.remove(future)
                return future
            maybe_log_progress()

    def complete_post_future(future: Any, *, wait_s: float) -> None:
        nonlocal compress_s, http_post_s, payload_count, posted_bytes, post_busy_s
        nonlocal post_wait_s, serialize_s
        (
            nbytes,
            busy_s,
            serialize_elapsed,
            compress_elapsed,
            http_post_elapsed,
            timing,
        ) = future.result()
        post_wait_s = post_wait_s + wait_s
        post_busy_s = post_busy_s + busy_s
        serialize_s = serialize_s + serialize_elapsed
        compress_s = compress_s + compress_elapsed
        http_post_s = http_post_s + http_post_elapsed
        add_vllm_refit_receiver_timing(receiver_timing, timing)
        posted_bytes = posted_bytes + nbytes * len(urls)
        payload_count = payload_count + 1

    def drain_next_future(
        futures: deque[Any],
        complete: Callable[..., None],
    ) -> None:
        wait_started = time.perf_counter()
        complete(
            wait_for_next_future(futures),
            wait_s=time.perf_counter() - wait_started,
        )

    def drain_done_futures(
        futures: deque[Any],
        complete: Callable[..., None],
    ) -> None:
        while True:
            future = pop_done_future(futures)
            if future is None:
                return
            complete(future, wait_s=0.0)

    def prepare_chunk_payload(
        chunk: TensorBatch,
        *,
        allow_dense: bool,
    ) -> tuple[TensorPayload, bool] | None:
        is_delta, prepared = delta_tracker.prepare_sparse_delta_payload(
            chunk,
            target_device=None,
        )
        if not is_delta:
            if not allow_dense:
                raise RuntimeError(
                    "Async HTTP encode only supports sparse delta chunks."
                )
            dense_tensors = cast(TensorBatch, prepared)
            delta_tracker.snapshot_pending_full_sync_baseline(dense_tensors)
            return (dense_tensors, G_DENSE_TRANSPORT, []), True

        payload = cast(TensorPayload, prepared)
        _, transport, metadata = payload
        if transport != G_SPARSE_INDICES_TRANSPORT:
            raise RuntimeError(f"Unsupported vLLM HTTP refit transport: {transport!r}")
        return (payload, False) if metadata else None

    def encode_chunk_payload(
        chunk: TensorBatch,
    ) -> tuple[tuple[TensorPayload, bool] | None, float]:
        started = time.perf_counter()
        return (
            prepare_chunk_payload(chunk, allow_dense=False),
            time.perf_counter() - started,
        )

    def complete_encoded_future(future: Any, *, wait_s: float) -> None:
        nonlocal encode_s, encode_wait_s
        encode_wait_s = encode_wait_s + wait_s
        encoded, elapsed = future.result()
        encode_s = encode_s + elapsed
        submit_prepared_payload(encoded)

    def encode_or_submit_chunk(chunk: TensorBatch) -> None:
        nonlocal encode_s
        if async_encode:
            assert encode_executor is not None
            while len(encode_inflight) >= encode_workers:
                drain_next_future(encode_inflight, complete_encoded_future)
            encode_inflight.append(encode_executor.submit(encode_chunk_payload, chunk))
            return
        started = time.perf_counter()
        prepared = prepare_chunk_payload(chunk, allow_dense=True)
        encode_s = encode_s + (time.perf_counter() - started)
        submit_prepared_payload(prepared)

    def submit_payload(payload: TensorPayload) -> None:
        nonlocal d2h_s
        started = time.perf_counter()
        while len(inflight) >= window:
            drain_next_future(inflight, complete_post_future)
        inflight.append(executor.submit(post_payload, payload))
        d2h_s = d2h_s + (time.perf_counter() - started)

    def maybe_log_progress() -> None:
        nonlocal last_progress_s
        now_s = time.perf_counter()
        if now_s - last_progress_s < progress_interval_s:
            return
        print(
            "REFIT_HTTP_PROGRESS "
            f"chunks={chunk_count} "
            f"submitted_payloads={payload_count + len(inflight)} "
            f"completed_payloads={payload_count} "
            f"queued_payloads={len(encode_inflight)} "
            f"posted_mb={posted_bytes / 1e6:.1f} "
            f"seconds={now_s - stream_start:.3f}",
            flush=True,
        )
        last_progress_s = now_s

    def submit_prepared_payload(prepared: tuple[TensorPayload, bool] | None) -> None:
        nonlocal has_dense_payload
        if prepared is None:
            return
        payload, is_dense = prepared
        submit_payload(payload)
        if is_dense:
            has_dense_payload = True
        maybe_log_progress()

    try:
        while True:
            started = time.perf_counter()
            chunk, pending_item = next_chunk(
                tensor_iterator,
                export_chunk_size,
                pending_item=pending_item,
            )
            export_pull_s += time.perf_counter() - started
            if not chunk:
                break
            chunk_count += 1
            encode_or_submit_chunk(chunk)
            drain_done_futures(encode_inflight, complete_encoded_future)
            drain_done_futures(inflight, complete_post_future)
            maybe_log_progress()
        while encode_inflight:
            drain_next_future(encode_inflight, complete_encoded_future)
        while inflight:
            drain_next_future(inflight, complete_post_future)
        if payload_count and (has_dense_payload or delta_tracker.async_receiver_apply):
            started = time.perf_counter()
            add_vllm_refit_receiver_timing(
                receiver_timing,
                flush_vllm_refit_urls(
                    urls,
                    api_key_env_var=api_key_env_var,
                    timeout_s=timeout_s,
                ),
            )
            flush_wait_s = time.perf_counter() - started
        delta_tracker.on_sync_succeeded()
    except Exception:
        for future in encode_inflight:
            with suppress(Exception):
                future.result()
        for future in inflight:
            with suppress(Exception):
                future.result()
        delta_tracker.on_sync_failed()
        raise

    total_s = time.perf_counter() - stream_start
    timing: dict[str, float | int] = {
        "total_s": total_s,
        "export_pull_s": export_pull_s,
        "encode_s": encode_s,
        "encode_wait_s": encode_wait_s,
        "d2h_s": d2h_s,
        "post_wait_s": post_wait_s,
        "post_busy_s": post_busy_s,
        "serialize_s": serialize_s,
        "compress_s": compress_s,
        "http_post_s": http_post_s,
        "flush_wait_s": flush_wait_s,
        "payloads": payload_count,
        "chunks": chunk_count,
        "posted_mb": posted_bytes / 1e6,
        "window": window,
        "export_chunk_mb": export_chunk_size / 1e6,
        "encode_workers": encode_workers,
    }
    timing_keys = tuple(timing)
    timing.update(receiver_timing)
    timing_parts = " ".join(
        f"{key}={_format_http_timing_value(key, timing[key])}"
        for key in (*timing_keys, *sorted(receiver_timing))
    )
    print(f"REFIT_HTTP_TIMING {timing_parts}", flush=True)
    return {
        "ok": True,
        "payloads": payload_count,
    }


def post_sparse_delta_payload_to_urls(
    base_urls: Sequence[str],
    body: bytes,
    *,
    api_key_env_var: str | None,
    timeout_s: float,
    compress_body: bool = True,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    endpoint_urls = [
        vllm_refit_sparse_delta_url(url)
        for url in normalize_vllm_refit_base_urls(base_urls)
    ]
    if not endpoint_urls:
        raise ValueError("At least one vLLM HTTP refit URL is required.")
    request_headers = extra_headers
    if compress_body:
        body, request_headers = encode_vllm_refit_request_body(body)
    responses = _post_refit_body_to_endpoint_urls(
        endpoint_urls,
        body,
        api_key_env_var=api_key_env_var,
        timeout_s=timeout_s,
        extra_headers=request_headers,
    )
    return {"ok": True, **_aggregate_fanout_refit_responses(responses)}


def _post_refit_body_to_endpoint_urls(
    endpoint_urls: Sequence[str],
    body: bytes | None,
    *,
    api_key_env_var: str | None,
    timeout_s: float,
    content_type: str = "application/octet-stream",
    extra_headers: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    if not endpoint_urls:
        raise ValueError("At least one vLLM HTTP refit URL is required.")
    responses = _map_parallel(
        "refit-fanout",
        endpoint_urls,
        lambda url: _http_request_json(
            url,
            body,
            api_key_env_var=api_key_env_var,
            timeout_s=timeout_s,
            content_type=content_type,
            extra_headers=extra_headers,
        ),
        _http_fanout_parallelism(len(endpoint_urls)),
    )
    for url, response in zip(endpoint_urls, responses, strict=False):
        if not response.get("ok", False):
            raise RuntimeError(f"vLLM HTTP sparse refit failed for {url}: {response}")
    return responses


def flush_vllm_refit_urls(
    base_urls: Sequence[str],
    *,
    api_key_env_var: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    endpoint_urls = [
        vllm_refit_flush_url(url) for url in normalize_vllm_refit_base_urls(base_urls)
    ]
    responses = _post_refit_body_to_endpoint_urls(
        endpoint_urls,
        b"{}",
        api_key_env_var=api_key_env_var,
        timeout_s=timeout_s,
        content_type="application/json",
    )
    result = {"ok": True, "urls": len(endpoint_urls)}
    result.update(_aggregate_fanout_refit_responses(responses))
    return result


def _aggregate_fanout_refit_responses(
    responses: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    payloads = 0
    for response in responses:
        if isinstance(response.get("payloads"), int):
            payloads += response["payloads"]
        for key, value in response.items():
            if (
                (key == "seconds" and isinstance(value, (int, float)))
                or is_refit_receiver_timing(key, value)
            ) and not isinstance(value, bool):
                result[key] = max(result.get(key, 0.0), float(value))
    if payloads:
        result["payloads"] = payloads
    return result


def add_vllm_refit_receiver_timing(
    result: dict[str, Any],
    timing: Mapping[str, Any],
) -> None:
    for key, value in timing.items():
        if is_refit_receiver_timing(key, value):
            result[key] = float(result.get(key, 0.0)) + float(value)


def start_vllm_refit_relay_server(
    base_urls: Sequence[str],
    *,
    host: str = "0.0.0.0",
    port: int | None = None,
    api_key_env_var: str | None = None,
    timeout_s: float = 600.0,
    advertised_host: str | None = None,
) -> tuple[threading.Thread, str, Any]:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from starlette.requests import ClientDisconnect

    urls = normalize_vllm_refit_base_urls(base_urls)
    if not urls:
        raise ValueError("At least one local vLLM refit URL is required.")
    port = _get_free_port_local() if port is None else int(port)
    app = FastAPI()

    def token_is_valid(raw_request: Request) -> bool:
        return vllm_refit_api_key_is_valid(api_key_env_var, raw_request.headers)

    def error_response(error: str, status_code: int) -> JSONResponse:
        return JSONResponse(
            {"ok": False, "error": error, "urls": len(urls)}, status_code
        )

    async def relay_health(raw_request) -> JSONResponse:
        if not token_is_valid(raw_request):
            return error_response("unauthorized", 403)
        try:
            await asyncio.to_thread(
                check_vllm_refit_health,
                urls,
                api_key_env_var=api_key_env_var,
                timeout_s=min(timeout_s, 30.0),
            )
        except Exception as exc:
            return error_response(str(exc), 503)
        return JSONResponse({"ok": True, "urls": len(urls)})

    async def relay_sparse_delta(raw_request) -> JSONResponse:
        if not token_is_valid(raw_request):
            return error_response("unauthorized", 403)
        try:
            body = await raw_request.body()
        except ClientDisconnect:
            return error_response(
                "Client disconnected while uploading sparse payload", 499
            )
        try:
            mode = _refit_request_body_encoding(raw_request.headers)
        except ValueError as exc:
            return error_response(str(exc), 400)
        forward_headers = {}
        if mode is not None:
            forward_headers["content-encoding"] = mode
            uncompressed_bytes = raw_request.headers.get(
                G_VLLM_REFIT_UNCOMPRESSED_BYTES_HEADER
            ) or raw_request.headers.get("X-Nemo-Rl-Refit-Uncompressed-Bytes")
            if uncompressed_bytes is not None:
                forward_headers[G_VLLM_REFIT_UNCOMPRESSED_BYTES_HEADER] = (
                    uncompressed_bytes
                )
        try:
            result = await asyncio.to_thread(
                post_sparse_delta_payload_to_urls,
                urls,
                body,
                api_key_env_var=api_key_env_var,
                timeout_s=timeout_s,
                compress_body=False,
                extra_headers=forward_headers,
            )
        except Exception as exc:
            return error_response(str(exc), 500)
        return JSONResponse(result)

    async def relay_flush(raw_request) -> JSONResponse:
        if not token_is_valid(raw_request):
            return error_response("unauthorized", 403)
        try:
            result = await asyncio.to_thread(
                flush_vllm_refit_urls,
                urls,
                api_key_env_var=api_key_env_var,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            return error_response(str(exc), 500)
        return JSONResponse(result)

    async def relay_generate(raw_request) -> JSONResponse:
        if not token_is_valid(raw_request):
            return error_response("unauthorized", 403)
        try:
            payload = json.loads((await raw_request.body()).decode("utf-8"))
            result = await asyncio.to_thread(
                post_generation_payload_to_urls,
                urls,
                payload,
                api_key_env_var=api_key_env_var,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            return error_response(str(exc), 500)
        return JSONResponse(result)

    for handler in (relay_health, relay_sparse_delta, relay_flush, relay_generate):
        handler.__annotations__["raw_request"] = Request
    app.add_api_route(G_VLLM_REFIT_HEALTH_PATH, relay_health, methods=["GET"])
    app.add_api_route(
        G_VLLM_REFIT_SPARSE_DELTA_PATH, relay_sparse_delta, methods=["POST"]
    )
    app.add_api_route(G_VLLM_REFIT_FLUSH_PATH, relay_flush, methods=["POST"])
    app.add_api_route(G_VLLM_GENERATE_PATH, relay_generate, methods=["POST"])

    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, timeout_keep_alive=120)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    relay_host = advertised_host or _get_node_ip_local()
    return thread, f"http://{relay_host}:{port}", server


def _http_request_json(
    url: str,
    body: bytes | None,
    *,
    api_key_env_var: str | None,
    timeout_s: float,
    content_type: str = "application/octet-stream",
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    headers = {"content-type": content_type}
    if extra_headers:
        headers.update(extra_headers)
    headers.update(vllm_refit_api_key_headers(api_key_env_var))
    session = _get_keepalive_session()
    if session is False:
        raise RuntimeError("The requests package is required for vLLM HTTP refit.")
    response = (
        session.get(url, headers=headers, timeout=timeout_s)
        if body is None
        else session.post(url, data=body, headers=headers, timeout=timeout_s)
    )
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code} from {url}: {response.text}")
    return {} if not response.text else json.loads(response.text)


def _require_zstandard():
    try:
        import zstandard
    except ImportError:
        raise RuntimeError(
            "vLLM HTTP refit body compression 'zstd' requires the zstandard package."
        ) from None
    return zstandard


def _zstd_compress(raw: bytes) -> bytes:
    threads = env_int(G_VLLM_REFIT_HTTP_ZSTD_THREADS_ENV, default=0, min_value=0)
    compressor = getattr(_HTTP_SESSION_LOCAL, "zstd_compressor", None)
    if (
        compressor is None
        or getattr(_HTTP_SESSION_LOCAL, "zstd_compressor_threads", None) != threads
    ):
        kwargs = {"level": 1}
        if threads:
            kwargs["threads"] = threads
        compressor = _require_zstandard().ZstdCompressor(**kwargs)
        _HTTP_SESSION_LOCAL.zstd_compressor = compressor
        _HTTP_SESSION_LOCAL.zstd_compressor_threads = threads
    return compressor.compress(raw)


def _zstd_decompress(raw: bytes) -> bytes:
    decompressor = getattr(_HTTP_SESSION_LOCAL, "zstd_decompressor", None)
    if decompressor is None:
        decompressor = _require_zstandard().ZstdDecompressor()
        _HTTP_SESSION_LOCAL.zstd_decompressor = decompressor
    return decompressor.decompress(raw)
