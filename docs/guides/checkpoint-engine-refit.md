# Checkpoint-Engine Refit

Checkpoint-engine refit updates non-colocated generation workers from policy
weights through a pluggable transfer backend. The built-in backend is NIXL,
which can use UCX/RDMA for large policy-to-vLLM generation refits.

Use this path when generation runs on dedicated resources:

- `policy.generation.colocated.enabled=false`
- `policy.generation.backend=vllm`
- `policy.generation.checkpoint_engine.enabled=true`

For colocated generation, NeMo RL continues to use the colocated IPC/HTTP refit
paths. For non-colocated generation without checkpoint-engine refit, NeMo RL
uses the NCCL collective update path.

## Enable NIXL Refit

Add a `checkpoint_engine` block under `policy.generation`:

```yaml
policy:
  generation:
    backend: vllm
    colocated:
      enabled: false
      resources:
        num_nodes: 1
        gpus_per_node: 8
    checkpoint_engine:
      enabled: true
      backend: nixl
      update_weights_bucket_megabytes: 2048
      engine_kwargs:
        nixl:
          device: cuda
          cleanup_after_load: false
          backend_name: UCX
          backend_init_params:
            ucx_error_handling_mode: none
            engine_config: MAX_RMA_RAILS=8
            device_list: "mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8"
```

`backend` selects the checkpoint-engine transfer backend. `nixl` is built in.
External backends can use a `module:ClassName` class path; see
[Checkpoint Engine Design](../design-docs/checkpoint-engines.md).

`update_weights_bucket_megabytes` controls the reusable transfer-buffer size on
every participating worker. Larger buckets reduce per-bucket overhead but
reserve more memory. `2048` MiB is the latest tested setting for the 30B MoE
CUDA-buffer run.

`engine_kwargs.<backend>` is passed to the backend constructor. Current NIXL
settings are:

| Key | Meaning |
|---|---|
| `device` | Transfer-buffer device. Use `cuda` for the GPU-buffer path when NIXL/UCX can register CUDA memory. Use `cpu` as a host-pinned fallback. |
| `cleanup_after_load` | Whether vLLM runs garbage collection and `torch.cuda.empty_cache()` after loading a refit. `false` avoids extra steady-state overhead when memory is stable. |
| `backend_name` | NIXL backend plugin name, usually `UCX`. |
| `backend_init_params` | Optional NIXL backend initialization parameters. Values are converted to strings before NIXL receives them. |

The current NIXL backend uses paired policy-to-rollout transfer. Allocate at
least as many policy workers as rollout workers.

Policy-side checkpoint-engine refit is wired for Megatron policy workers and
DTensor/FSDP2 policy workers. FP8 KV-cache scale transfer is still implemented
only by the Megatron policy path; DTensor workers reject `kv_scales` just as
they do on the NCCL refit path.

## Runtime Requirements

NIXL must be importable in every worker environment that participates in refit:

- policy worker environments
- vLLM worker environments, including async vLLM worker environments
- the driver/base environment if the launcher performs NIXL preflight checks

Install NIXL in those environments or bake it into the container image. For CUDA
12 environments this is typically:

```sh
uv pip install nixl-cu12 nixl
```

Keep checkpoint-engine feature selection in YAML/config. Use environment
variables only for UCX/NIXL runtime transport selection:

```sh
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1
export UCX_TLS=rc,cuda_copy,cuda_ipc,self,sm
export UCX_IB_ROCE_REACHABILITY_MODE=all
export UCX_MAX_RNDV_RAILS=8
export UCX_WARN_UNUSED_ENV_VARS=n
export NIXL_LOG_LEVEL=INFO
```

When vLLM starts nested Ray workers, make sure transport variables are copied
into those workers:

```sh
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=MELLANOX_
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=LD_LIBRARY_PATH,NIXL_LOG_LEVEL,NVIDIA_VISIBLE_DEVICES,UCX_NET_DEVICES,UCX_TLS,UCX_IB_ROCE_REACHABILITY_MODE,UCX_MAX_RNDV_RAILS,UCX_WARN_UNUSED_ENV_VARS
```

Use NIC names that exist on the target nodes. With `UCX_LOG_LEVEL=info`, UCX
should report an RDMA transport such as `rc_mlx5`. If UCX reports TCP-only
transport, refit will be much slower.

## Fault Tolerance

NIXL/UCX can help a job fail fast on transport errors, which lets the outer
launcher restart from the latest durable checkpoint. The current
checkpoint-engine refit path does not transparently replace a dead Ray actor or
rebuild the vLLM generation group inside the same training step.

For production or fault-injection runs, prefer UCX peer error handling:

```yaml
policy:
  generation:
    checkpoint_engine:
      enabled: true
      backend: nixl
      engine_kwargs:
        nixl:
          backend_name: UCX
          backend_init_params:
            ucx_error_handling_mode: peer
```

`ucx_error_handling_mode: none` is useful for stable performance experiments,
but it gives UCX less ability to report failed peers to NIXL.

Pair UCX peer error handling with bounded retry and keepalive values:

```sh
export UCX_RC_TIMEOUT=30s
export UCX_RC_RETRY_COUNT=7
export UCX_KEEPALIVE_INTERVAL=1s
export UCX_KEEPALIVE_NUM_EPS=10
```

These settings do not recover an individual transfer. They bound how long UCX
waits before declaring a peer unhealthy. Use normal NeMo RL checkpointing for
restartable training:

```yaml
checkpointing:
  enabled: true
  checkpoint_dir: /path/to/restartable/checkpoints
```

## Command-Line Override Example

The same settings can be passed as Hydra overrides:

```sh
uv run --extra mcore --extra vllm examples/run_grpo.py \
  --config examples/configs/grpo_math_8B_megatron.yaml \
  cluster.num_nodes=2 \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=1 \
  policy.generation.colocated.resources.gpus_per_node=8 \
  policy.generation.checkpoint_engine.enabled=true \
  policy.generation.checkpoint_engine.backend=nixl \
  policy.generation.checkpoint_engine.update_weights_bucket_megabytes=2048 \
  ++policy.generation.checkpoint_engine.engine_kwargs.nixl.device=cuda \
  ++policy.generation.checkpoint_engine.engine_kwargs.nixl.cleanup_after_load=false \
  ++policy.generation.checkpoint_engine.engine_kwargs.nixl.backend_name=UCX \
  ++policy.generation.checkpoint_engine.engine_kwargs.nixl.backend_init_params.ucx_error_handling_mode=none
```

Adjust `cluster.num_nodes` and
`policy.generation.colocated.resources.{num_nodes,gpus_per_node}` so the cluster
has enough policy and generation resources. On two 8-GPU nodes, the snippet
above dedicates one node to vLLM generation and leaves one node for policy
workers.

## Verified 30B MoE Result

The latest two-node Qwen3-30B-A3B NIXL run used the NeMo RL container on the
Slurm `batch` partition and completed successfully:

```text
Job: 12980977
Model: Qwen3-30B-A3B
Nodes: 2 x 8 H100
Transfer buffers: cuda
Bucket size: 2048 MiB
NIXL backend: UCX
Slurm state: COMPLETED, exit 0:0
```

The driver log showed:

```text
Using checkpoint-engine refit backend: nixl
Total setup time: 247.9s
[vLLM refit] Loaded 18867 tensors in 29 batches via checkpoint engine; bytes=56.87GiB total=1.94s receive=1.45s load=0.49s
prepare_for_generation/transfer_and_update_weights: 3.56s (5.0%)
Total step time: 70.99s
```

Other vLLM workers repeated the refit line at about `1.93s`. The full
`transfer_and_update_weights` timer includes the GRPO orchestration window,
while the `[vLLM refit]` line measures receive plus vLLM load time inside the
vLLM worker.

## Verify a Run

The driver log should show:

```text
Using checkpoint-engine refit backend: nixl
```

During each update, vLLM prints checkpoint-engine load timing:

```text
[vLLM refit] Loaded ... via checkpoint engine; bytes=... total=... receive=... load=...
```

With `UCX_LOG_LEVEL=info`, UCX should show an RDMA transport such as:

```text
rma(rc_mlx5/mlx5_0:1)
```

If UCX reports only TCP transports, inspect `UCX_NET_DEVICES`, `UCX_TLS`,
container device visibility, and the network interfaces available on every
node.

## Try a Correctness Smoke Test

`tools/refit_verifier.py` compares vLLM and Megatron logprobs after a refit:

```sh
uv run --extra mcore --extra vllm python tools/refit_verifier.py \
  --model_name /path/to/model \
  --tp_size 1 \
  --ep_size 1 \
  --pp_size 1
```

This tool is useful for validating refit correctness and model compatibility.
It currently exercises the colocated refit path. To test the NIXL
checkpoint-engine path, run a non-colocated GRPO job with
`policy.generation.checkpoint_engine.enabled=true` and inspect the log markers
above.

## Troubleshooting

### The run errors with "checkpoint-engine refit is only supported for non-colocated generation"

Set:

```yaml
policy:
  generation:
    colocated:
      enabled: false
```

Checkpoint-engine refit is for non-colocated generation only.

### NIXL cannot be imported

Install NIXL in the environment that failed. In Ray runs, policy workers, vLLM
workers, and async vLLM workers may use different virtual environments.

### UCX logs say CUDA support was not found

This is expected when `engine_kwargs.nixl.device=cpu`, because transfer buffers
live in host memory. If you set `device=cuda`, the NIXL/UCX build must support
CUDA memory registration.

### NIXL works but is slow

Check the UCX transport line. RDMA should show `rc_mlx5` or another expected
RDMA transport. If it shows TCP only, verify `UCX_NET_DEVICES`, `UCX_TLS`,
container device visibility, and network interface availability on every node.

Also check bucket sizing. Very small buckets increase metadata and
synchronization overhead. Start with `update_weights_bucket_megabytes=2048` for
large models and adjust only after measuring.

### A node or NIC failure causes the job to hang

Use `backend_init_params.ucx_error_handling_mode=peer` and set bounded UCX
retry and keepalive values. Without UCX peer error handling, some transport
failures can look like an indefinitely pending NIXL transfer rather than a clean
`ERR` state.
