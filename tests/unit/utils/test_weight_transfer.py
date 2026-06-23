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

import io
import threading

import pytest
import torch

from nemo_rl.models.generation.vllm.config import VllmDeltaCompressionConfig
from nemo_rl.utils import weight_transfer_http as http_utils
from nemo_rl.utils import weight_transfer_sparse_codec as sparse_codec
from nemo_rl.utils.weight_transfer import (
    packed_weight_transfer_consumer,
    packed_weight_transfer_producer,
)
from nemo_rl.utils.weight_transfer_delta_tracker import (
    DeltaCompressionTracker,
    create_vllm_delta_transfer_tracker,
)
from nemo_rl.utils.weight_transfer_http import (
    G_VLLM_REFIT_API_KEY_HEADER,
    G_VLLM_REFIT_UNCOMPRESSED_BYTES_HEADER,
    decode_vllm_refit_request_body,
    encode_vllm_refit_request_body,
    init_sparse_delta_baseline_from_iterator,
    post_sparse_delta_payload_to_urls,
    stream_sparse_delta_payloads_via_http,
    vllm_refit_api_key_headers,
    vllm_refit_api_key_is_valid,
)
from nemo_rl.utils.weight_transfer_protocol import (
    G_DELTA_UPDATE_KIND,
    G_DENSE_TRANSPORT,
    G_INDEX_END_KEY,
    G_INDEX_START_KEY,
    G_PACKED_INDICES_NAME,
    G_PACKED_VALUES_NAME,
    G_SPARSE_INDICES_TRANSPORT,
    G_TRANSFER_DONE_KIND,
    additive_weight_load_context,
    broadcast_header,
    pack_named_tensors,
    unpack_named_tensors,
)


class _NoopGroup:
    rank = 0

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        del tensor, src


class _QueuedBroadcastGroup:
    def __init__(self, rank: int, queue: list[torch.Tensor]) -> None:
        self.rank = rank
        self._queue = queue

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        if self.rank == src:
            self._queue.append(tensor.detach().cpu().clone())
            return
        queued = self._queue.pop(0)
        tensor.copy_(queued.to(device=tensor.device, dtype=tensor.dtype))


def _tracker(
    full_sync_interval: int = 3,
    index_encoding: str = "indices",
) -> DeltaCompressionTracker:
    return DeltaCompressionTracker(
        {
            "full_sync_interval": full_sync_interval,
            "sparse_bucket_size_bytes": 1024,
            "dtype": "float32",
            "index_encoding": index_encoding,
        }
    )


def _in_memory_tracker(monkeypatch, **kwargs) -> DeltaCompressionTracker:
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    return _tracker(**kwargs)


def _commit_initial_baseline(
    tracker: DeltaCompressionTracker,
    tensors: list[tuple[str, torch.Tensor]],
) -> None:
    is_delta, payload = tracker.prepare_sparse_delta_payload(tensors)
    assert not is_delta
    tracker.snapshot_pending_full_sync_baseline(payload)
    tracker.on_sync_succeeded()


def _decode_sparse_payload(payload_tensors, metadata) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for batch in sparse_codec.decode_sparse(
            payload_tensors,
            metadata,
            device="cpu",
            byte_cap=1024,
        )
        for name, tensor in batch
    }


def _configure_http_sparse_stream_test(
    monkeypatch,
    *,
    encode_workers: int = 1,
) -> None:
    env = {
        "NRL_REFIT_BASELINE_IN_MEMORY": "1",
        "NRL_REFIT_HTTP_INFLIGHT_BUCKETS": "1",
        "NRL_REFIT_HTTP_EXPORT_CHUNK_BYTES": "4",
        "NRL_REFIT_HTTP_ENCODE_WORKERS": str(encode_workers),
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("NRL_REFIT_HTTP_BODY_COMPRESS", raising=False)


class _SparseDeltaHttpRecorder:
    def __init__(
        self,
        *,
        decode_posts: bool = True,
        post_response: list[dict] | None = None,
        flush_response: dict | None = None,
        post_hook=None,
    ) -> None:
        self.decode_posts = decode_posts
        self.post_response = post_response or []
        self.flush_response = flush_response or {
            "ok": True,
            "receiver_total_s": 0.0,
        }
        self.post_hook = post_hook
        self.posts = []
        self.flushes = []

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "nemo_rl.utils.weight_transfer_http._post_refit_body_to_endpoint_urls",
            self.post,
        )
        monkeypatch.setattr(
            "nemo_rl.utils.weight_transfer_http.flush_vllm_refit_urls",
            self.flush,
        )

    def post(
        self,
        endpoint_urls,
        body,
        *,
        api_key_env_var,
        timeout_s,
        content_type="application/octet-stream",
        extra_headers=None,
    ):
        del api_key_env_var, timeout_s, content_type
        if self.decode_posts:
            request = torch.load(io.BytesIO(body), weights_only=True)
        else:
            request = (list(endpoint_urls), body, extra_headers)
        if self.post_hook is not None:
            self.post_hook(request)
        self.posts.append(request)
        return self.post_response

    def flush(self, base_urls, *, api_key_env_var, timeout_s):
        self.flushes.append((list(base_urls), api_key_env_var, timeout_s))
        return self.flush_response


def test_pack_named_tensors_round_trips_mixed_dtypes() -> None:
    tensors = [
        ("weight", torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)),
        ("bias", torch.arange(3, dtype=torch.int32)),
    ]

    packed, entries = pack_named_tensors(tensors)
    unpacked = unpack_named_tensors(packed, entries)

    assert [name for name, _ in unpacked] == ["weight", "bias"]
    for (_, expected), (_, actual) in zip(tensors, unpacked, strict=True):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)


def test_additive_weight_load_context_handles_param_data_and_views() -> None:
    param = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    untouched = torch.zeros(2, dtype=torch.float32)

    with additive_weight_load_context([param]):
        param.data.copy_(torch.ones_like(param))
        param.data.view(-1).narrow(0, 2, 2).copy_(torch.tensor([10.0, 20.0]))
        untouched.copy_(torch.ones_like(untouched))

    torch.testing.assert_close(
        param,
        torch.tensor([[1.0, 2.0, 13.0], [24.0, 5.0, 6.0]]),
    )
    torch.testing.assert_close(untouched, torch.ones_like(untouched))


def test_merge_sparse_payloads_offsets_metadata() -> None:
    payload_a = sparse_codec.encode_sparse_infos(
        [
            (
                "a",
                torch.zeros(4, dtype=torch.float32),
                torch.tensor([1, 3], dtype=torch.int64),
                torch.tensor([0.5, 0.75], dtype=torch.float32),
            )
        ],
    )
    payload_b = sparse_codec.encode_sparse_infos(
        [
            (
                "b",
                torch.zeros(3, dtype=torch.float32),
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([2.0], dtype=torch.float32),
            )
        ],
    )

    tensors, _, metadata = sparse_codec.merge_sparse_payloads([payload_a, payload_b])
    packed = dict(tensors)

    assert torch.equal(
        packed[G_PACKED_INDICES_NAME], torch.tensor([1, 3], dtype=torch.int32)
    )
    assert torch.equal(packed[G_PACKED_VALUES_NAME], torch.tensor([0.5, 0.75, 2.0]))
    assert metadata[0][G_INDEX_START_KEY] == 0
    assert metadata[0][G_INDEX_END_KEY] == 2
    assert metadata[1][G_INDEX_START_KEY] == 2
    assert metadata[1][G_INDEX_END_KEY] == 2
    assert metadata[1]["value_start"] == 2
    assert metadata[1]["value_end"] == 3


def test_sparse_indices_choose_width_from_flat_locations() -> None:
    small_payload_tensors, _transport, small_metadata = (
        sparse_codec.encode_sparse_infos(
            [
                (
                    "linear.weight",
                    torch.zeros(8, dtype=torch.float32),
                    torch.tensor([1, 3], dtype=torch.int64),
                    torch.tensor([2.0, 4.0], dtype=torch.float32),
                )
            ],
            index_encoding="indices",
        )
    )
    large_location = 2**31 + 5
    # Use a non-contiguous, multi-element location set so the explicit-indices
    # path is exercised; a single location collapses to a range encoding.
    large_payload_tensors, _transport, large_metadata = (
        sparse_codec.encode_sparse_infos(
            [
                (
                    "huge.weight",
                    torch.empty(0, dtype=torch.float32),
                    torch.tensor([0, large_location], dtype=torch.int64),
                    torch.tensor([6.0, 7.0], dtype=torch.float32),
                )
            ],
            index_encoding="indices",
        )
    )
    small_packed = dict(small_payload_tensors)[G_PACKED_INDICES_NAME]
    large_packed = dict(large_payload_tensors)[G_PACKED_INDICES_NAME]

    assert small_packed.dtype == torch.int32
    assert small_metadata[0]["index_encoding"] == "indices"
    assert small_metadata[0]["explicit_index_width"] == 4
    assert large_packed.dtype == torch.int64
    assert large_metadata[0]["explicit_index_width"] == 8
    decoded = sparse_codec.sparse_locations_for_item(
        large_metadata[0],
        large_packed,
        (0, 2, 0, 2),
        device="cpu",
    )
    assert decoded.tolist() == [0, large_location]

    merged_tensors, _transport, merged_metadata = sparse_codec.merge_sparse_payloads(
        [
            (small_payload_tensors, G_SPARSE_INDICES_TRANSPORT, small_metadata),
            (large_payload_tensors, G_SPARSE_INDICES_TRANSPORT, large_metadata),
        ]
    )
    merged_packed = dict(merged_tensors)[G_PACKED_INDICES_NAME]

    assert merged_packed.dtype == torch.int64
    assert [item["explicit_index_width"] for item in merged_metadata] == [8, 8]


def test_collective_consumer_decodes_non_source_sparse_delta_header(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    queue: list[torch.Tensor] = []
    source = _QueuedBroadcastGroup(rank=0, queue=queue)
    receiver = _QueuedBroadcastGroup(rank=1, queue=queue)
    payload_tensors, transport, metadata = sparse_codec.encode_sparse_infos(
        [
            (
                "linear.weight",
                torch.zeros(8, dtype=torch.float32),
                torch.tensor([1, 4], dtype=torch.int64),
                torch.tensor([2.0, 3.0], dtype=torch.float32),
            )
        ],
        index_encoding="deltas",
    )
    packed_payload, payload_entries = pack_named_tensors(payload_tensors)
    assert transport == G_SPARSE_INDICES_TRANSPORT
    header = {
        "kind": G_DELTA_UPDATE_KIND,
        "transport": transport,
        "payload_entries": payload_entries,
        "payload_numel": int(packed_payload.numel()),
        "sparse_metadata": metadata,
        "is_delta_sync": True,
    }

    broadcast_header(header, group=source, src=0, device="cpu")
    source.broadcast(packed_payload, src=0)
    broadcast_header({"kind": G_TRANSFER_DONE_KIND}, group=source, src=0, device="cpu")
    loaded_sparse: list[tuple[list[tuple[str, torch.Tensor]], list[dict]]] = []

    result = packed_weight_transfer_consumer(
        group=receiver,
        src=0,
        load_full_weights_func=lambda tensors: pytest.fail(
            f"unexpected full payload: {tensors}"
        ),
        load_sparse_weights_func=lambda tensors, sparse_metadata: loaded_sparse.append(
            (tensors, sparse_metadata)
        ),
        device="cpu",
    )

    assert result.loaded_any
    assert result.is_delta_sync
    assert not queue
    assert len(loaded_sparse) == 1
    loaded_tensors, loaded_metadata = loaded_sparse[0]
    assert loaded_metadata[0]["index_encoding"] == "deltas"
    assert loaded_metadata[0]["name"] == "linear.weight"
    decoded = _decode_sparse_payload(loaded_tensors, loaded_metadata)
    torch.testing.assert_close(
        decoded["linear.weight"],
        torch.tensor([0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]),
    )


def test_collective_full_sync_defers_source_baseline_prewarm(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("NRL_REFIT_CPU_TARGET_PACKED_TENSOR_SIZE", "1024")
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    tracker = _tracker()
    tensor = torch.tensor([1.0, 2.0, 3.0])

    packed_weight_transfer_producer(
        [("linear.weight", tensor)],
        group=_NoopGroup(),
        src=0,
        delta_tracker=tracker,
    )

    assert tracker.committed_syncs == 0
    assert tracker.has_pending_full_sync_baseline()
    tracker.snapshot_pending_full_sync_baseline([("linear.weight", tensor)])
    tracker.on_sync_succeeded()
    assert tracker.committed_syncs == 1
    assert torch.equal(tracker.baseline["linear.weight"], tensor)


def test_collective_full_sync_snapshots_baseline_without_prewarm(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("NRL_REFIT_CPU_TARGET_PACKED_TENSOR_SIZE", "1024")
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    monkeypatch.setenv("NRL_REFIT_PREWARM_DELTA_BASELINE", "0")
    tracker = _tracker()
    tensor = torch.tensor([1.0, 2.0, 3.0])

    packed_weight_transfer_producer(
        [("linear.weight", tensor)],
        group=_NoopGroup(),
        src=0,
        delta_tracker=tracker,
    )

    assert tracker.committed_syncs == 1
    assert not tracker.has_pending_full_sync_baseline()
    assert tracker.is_delta_sync()
    torch.testing.assert_close(tracker.baseline["linear.weight"], tensor)

    tensor.add_(torch.tensor([0.0, 4.0, 0.0]))
    packed_weight_transfer_producer(
        [("linear.weight", tensor)],
        group=_NoopGroup(),
        src=0,
        delta_tracker=tracker,
    )

    assert tracker.committed_syncs == 2
    torch.testing.assert_close(tracker.baseline["linear.weight"], tensor)

    packed_weight_transfer_producer(
        [("linear.weight", tensor)],
        group=_NoopGroup(),
        src=0,
        delta_tracker=tracker,
    )

    assert tracker.committed_syncs == 3
    torch.testing.assert_close(tracker.baseline["linear.weight"], tensor)


def test_delta_tracker_accepts_pydantic_delta_config(monkeypatch) -> None:
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    tracker = create_vllm_delta_transfer_tracker(
        {
            "delta_compression": VllmDeltaCompressionConfig(
                enabled=True,
                dtype="float32",
                full_sync_interval=3,
                sparse_bucket_size_bytes=1024,
                delta_load_batch_size_bytes=1024,
                index_encoding="deltas",
            )
        }
    )

    assert tracker is not None
    assert tracker.index_encoding == "deltas"


def test_delta_tracker_respects_periodic_full_sync_interval(monkeypatch) -> None:
    tracker = _in_memory_tracker(monkeypatch, full_sync_interval=3)
    tensor = torch.zeros(4, dtype=torch.float32)

    _commit_initial_baseline(tracker, [("linear.weight", tensor)])
    assert tracker.is_delta_sync()

    for value in (1.0, 2.0):
        tensor[0] = value
        is_delta, _payload = tracker.prepare_sparse_delta_payload(
            [("linear.weight", tensor)]
        )
        assert is_delta
        tracker.on_sync_succeeded()

    assert tracker.committed_syncs == 3
    assert not tracker.is_delta_sync()
    tensor[0] = 3.0
    is_delta, payload = tracker.prepare_sparse_delta_payload(
        [("linear.weight", tensor)]
    )

    assert not is_delta
    assert len(payload) == 1
    assert payload[0][0] == "linear.weight"
    assert payload[0][1] is tensor
    assert tracker.has_pending_full_sync_baseline()


def test_delta_tracker_scans_multi_tensor_delta_chunk(monkeypatch) -> None:
    tracker = _in_memory_tracker(monkeypatch, index_encoding="deltas")
    weight = torch.zeros(4, dtype=torch.float32)
    bias = torch.zeros(3, dtype=torch.float32)
    _commit_initial_baseline(
        tracker,
        [("linear.weight", weight), ("bias", bias)],
    )

    weight[1] = 2.0
    bias[0] = 3.0
    bias[2] = 4.0
    is_delta, payload = tracker.prepare_sparse_delta_payload(
        [("linear.weight", weight), ("bias", bias)]
    )

    assert is_delta
    payload_tensors, _transport, metadata = payload
    assert [item["name"] for item in metadata] == ["linear.weight", "bias"]
    decoded = _decode_sparse_payload(payload_tensors, metadata)
    torch.testing.assert_close(
        decoded["linear.weight"],
        torch.tensor([0.0, 2.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(decoded["bias"], torch.tensor([3.0, 0.0, 4.0]))

    tracker.on_sync_succeeded()
    torch.testing.assert_close(tracker.baseline["linear.weight"], weight)
    torch.testing.assert_close(tracker.baseline["bias"], bias)


def test_http_periodic_full_sync_posts_dense_payload_and_flushes(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    monkeypatch.setenv("NRL_REFIT_HTTP_INFLIGHT_BUCKETS", "1")
    monkeypatch.setattr(
        "nemo_rl.utils.weight_transfer_http.get_target_packed_tensor_size",
        lambda: 1024,
    )
    tracker = _tracker(full_sync_interval=2)
    tensor = torch.zeros(4, dtype=torch.float32)
    _commit_initial_baseline(tracker, [("linear.weight", tensor)])

    tensor[0] = 1.0
    is_delta, _payload = tracker.prepare_sparse_delta_payload(
        [("linear.weight", tensor)]
    )
    assert is_delta
    tracker.on_sync_succeeded()
    assert not tracker.is_delta_sync()

    http_recorder = _SparseDeltaHttpRecorder(
        flush_response={"ok": True, "receiver_total_s": 0.25},
    )
    http_recorder.install(monkeypatch)
    tensor[1] = 2.0

    result = stream_sparse_delta_payloads_via_http(
        [("linear.weight", tensor)],
        delta_tracker=tracker,
        is_payload_source=True,
        refit_urls=["http://worker"],
    )

    assert result["payloads"] == 1
    assert len(http_recorder.posts) == 1
    assert http_recorder.posts[0]["transport"] == G_DENSE_TRANSPORT
    assert http_recorder.posts[0]["metadata"] == []
    assert http_recorder.posts[0]["payload_tensors"][0][0] == "linear.weight"
    torch.testing.assert_close(http_recorder.posts[0]["payload_tensors"][0][1], tensor)
    assert http_recorder.flushes == [(["http://worker"], None, 600.0)]
    assert tracker.committed_syncs == 3
    torch.testing.assert_close(tracker.baseline["linear.weight"], tensor)


def test_http_refit_request_body_zlib_round_trips(monkeypatch) -> None:
    monkeypatch.setenv("NRL_REFIT_HTTP_BODY_COMPRESS", "zlib")
    body = (b"nemo-rl-refit-payload" * 1024) + torch.arange(32).numpy().tobytes()

    encoded, headers = encode_vllm_refit_request_body(body)

    assert headers["content-encoding"] == "zlib"
    assert headers[G_VLLM_REFIT_UNCOMPRESSED_BYTES_HEADER] == str(len(body))
    assert len(encoded) < len(body)
    assert decode_vllm_refit_request_body(encoded, headers) == body


def test_zstd_thread_envs_configure_compressors(monkeypatch) -> None:
    created = []

    class FakeCompressor:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def compress(self, raw):
            return raw

    class FakeZstandard:
        ZstdCompressor = FakeCompressor

    monkeypatch.setenv("NRL_REFIT_HTTP_ZSTD_THREADS", "4")
    monkeypatch.setenv("NRL_REFIT_SPARSE_INDEX_ZSTD_THREADS", "3")
    monkeypatch.setattr(http_utils, "_HTTP_SESSION_LOCAL", threading.local())
    monkeypatch.setattr(sparse_codec, "_ZSTD_LOCAL", threading.local())
    monkeypatch.setattr(http_utils, "_require_zstandard", lambda: FakeZstandard)
    monkeypatch.setattr(sparse_codec, "_require_zstandard", lambda: FakeZstandard)

    assert http_utils._zstd_compress(b"http") == b"http"
    assert sparse_codec._zstd_compress(b"indices") == b"indices"
    assert created == [
        {"level": 1, "threads": 4},
        {"level": 1, "threads": 3},
    ]


def test_http_sparse_post_forwards_compressed_body_and_receiver_timing(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NRL_REFIT_HTTP_FANOUT_WORKERS", "9")
    forward_headers = {
        "content-encoding": "zstd",
        G_VLLM_REFIT_UNCOMPRESSED_BYTES_HEADER: "15",
    }
    http_recorder = _SparseDeltaHttpRecorder(
        decode_posts=False,
        post_response=[
            {
                "ok": True,
                "payloads": 2,
                "receiver_decode_s": 0.25,
                "receiver_total_s": 1.0,
            },
            {
                "ok": True,
                "payloads": 3,
                "receiver_decode_s": 0.5,
                "receiver_total_s": 0.75,
            },
        ],
    )
    http_recorder.install(monkeypatch)

    result = post_sparse_delta_payload_to_urls(
        ["http://worker-a", "http://worker-b"],
        b"already-decoded",
        api_key_env_var=None,
        timeout_s=3.0,
        compress_body=False,
        extra_headers=forward_headers,
    )

    assert result["ok"]
    assert result["payloads"] == 5
    assert result["receiver_decode_s"] == 0.5
    assert result["receiver_total_s"] == 1.0
    assert http_recorder.posts == [
        (
            [
                "http://worker-a/nemo-rl/refit/sparse-delta",
                "http://worker-b/nemo-rl/refit/sparse-delta",
            ],
            b"already-decoded",
            forward_headers,
        )
    ]
    assert http_utils._http_fanout_parallelism(2) == 9


def test_vllm_refit_api_key_auth_modes(monkeypatch) -> None:
    assert vllm_refit_api_key_headers(None) == {}
    assert vllm_refit_api_key_is_valid(None, {})

    monkeypatch.delenv("NRL_TEST_REFIT_KEY", raising=False)
    with pytest.raises(RuntimeError, match="configured but unset or empty"):
        vllm_refit_api_key_headers("NRL_TEST_REFIT_KEY")
    assert not vllm_refit_api_key_is_valid("NRL_TEST_REFIT_KEY", {})

    monkeypatch.setenv("NRL_TEST_REFIT_KEY", "secret")
    assert vllm_refit_api_key_headers("NRL_TEST_REFIT_KEY") == {
        G_VLLM_REFIT_API_KEY_HEADER: "secret"
    }
    assert vllm_refit_api_key_is_valid(
        "NRL_TEST_REFIT_KEY",
        {G_VLLM_REFIT_API_KEY_HEADER: "secret"},
    )
    assert not vllm_refit_api_key_is_valid("NRL_TEST_REFIT_KEY", {})
    assert not vllm_refit_api_key_is_valid(
        "NRL_TEST_REFIT_KEY",
        {G_VLLM_REFIT_API_KEY_HEADER: "wrong"},
    )


def test_http_sparse_baseline_initializes_source_iterator(monkeypatch) -> None:
    _configure_http_sparse_stream_test(monkeypatch)
    tracker = _in_memory_tracker(monkeypatch)
    tensor_a = torch.tensor([1.0], dtype=torch.float32)
    tensor_b = torch.tensor([2.0], dtype=torch.float32)

    init_sparse_delta_baseline_from_iterator(
        iter([("a.weight", tensor_a), ("b.weight", tensor_b)]),
        delta_tracker=tracker,
        is_payload_source=True,
    )

    assert tracker.committed_syncs == 1
    assert tracker.is_delta_sync()
    torch.testing.assert_close(tracker.baseline["a.weight"], tensor_a)
    torch.testing.assert_close(tracker.baseline["b.weight"], tensor_b)


def test_http_sparse_stream_overlaps_export_encode_and_post(monkeypatch) -> None:
    _configure_http_sparse_stream_test(monkeypatch, encode_workers=2)
    tracker = _tracker(index_encoding="indices")
    tensors = {
        name: torch.zeros(1, dtype=torch.float32)
        for name in ("a.weight", "b.weight", "c.weight")
    }
    _commit_initial_baseline(tracker, list(tensors.items()))
    tensor_a = tensors["a.weight"]
    tensor_b = tensors["b.weight"]
    tensor_c = tensors["c.weight"]
    tensor_a.add_(1.0)
    tensor_b.add_(2.0)
    tensor_c.add_(3.0)

    original_prepare = tracker.prepare_sparse_delta_payload
    first_encode_started = threading.Event()
    first_post_started = threading.Event()
    second_chunk_read = threading.Event()
    third_chunk_read = threading.Event()
    allow_first_encode_finish = threading.Event()
    allow_first_post_finish = threading.Event()

    def delayed_prepare(chunk, *, target_device=None):
        if chunk[0][0] == "a.weight":
            first_encode_started.set()
            assert allow_first_encode_finish.wait(2.0)
        return original_prepare(
            chunk,
            target_device=target_device,
        )

    def on_post(_request):
        if not first_post_started.is_set():
            first_post_started.set()
            assert allow_first_post_finish.wait(2.0)

    def export_iter():
        yield ("a.weight", tensor_a)
        assert first_encode_started.wait(2.0)
        second_chunk_read.set()
        allow_first_encode_finish.set()
        yield ("b.weight", tensor_b)
        assert first_post_started.wait(2.0)
        third_chunk_read.set()
        allow_first_post_finish.set()
        yield ("c.weight", tensor_c)

    monkeypatch.setattr(tracker, "prepare_sparse_delta_payload", delayed_prepare)
    _SparseDeltaHttpRecorder(post_hook=on_post).install(monkeypatch)

    result = stream_sparse_delta_payloads_via_http(
        export_iter(),
        delta_tracker=tracker,
        is_payload_source=True,
        refit_urls=["http://worker"],
    )

    assert second_chunk_read.is_set()
    assert third_chunk_read.is_set()
    assert result["payloads"] == 3


def test_collective_delta_sync_rejects_dense_payload_during_delta(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("NRL_REFIT_CPU_TARGET_PACKED_TENSOR_SIZE", "1024")
    tracker = _tracker()
    tracker.committed_syncs = 1

    with pytest.raises(RuntimeError, match="dense payload during a delta sync"):
        packed_weight_transfer_producer(
            [("missing_baseline.weight", torch.ones(2))],
            group=_NoopGroup(),
            src=0,
            delta_tracker=tracker,
        )
