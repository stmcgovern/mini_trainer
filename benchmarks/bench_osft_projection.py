"""Benchmark: OSFT projection overhead as fraction of training step time.

Measures how much wall-clock time project_gradients and project_parameters
consume relative to the full training step (forward + backward + optimizer).
Uses torch.profiler record_function annotations added to osft_utils.py.

Usage:
  # 2-GPU distributed with Llama-8B shapes (default)
  CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
      benchmarks/bench_osft_projection.py

  # Single GPU with small model
  CUDA_VISIBLE_DEVICES=6 python benchmarks/bench_osft_projection.py --small

  # Control model size
  benchmarks/bench_osft_projection.py --hidden-size 4096 --num-layers 32
"""

import argparse
import os
import time

os.environ["TESTING"] = "true"

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import LlamaConfig, LlamaForCausalLM

from mini_trainer.none_reduction_losses import hf_fixed_cross_entropy_none_reduction
from mini_trainer.setup_model_for_training import setup_model, setup_training_components
from mini_trainer.utils import patch_target_module


LLAMA_8B_SHAPES_CONFIG = dict(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=4,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=4096,
    rope_theta=500000.0,
    hidden_act="silu",
)

SMALL_CONFIG = dict(
    vocab_size=1000,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    rope_theta=10000.0,
    hidden_act="silu",
)


def create_model(tmp_dir, config_overrides=None):
    cfg = dict(LLAMA_8B_SHAPES_CONFIG)
    if config_overrides:
        cfg.update(config_overrides)

    config = LlamaConfig(**cfg)
    model = LlamaForCausalLM(config)
    model.save_pretrained(tmp_dir)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.save_pretrained(tmp_dir)
    return tmp_dir, cfg


def setup_dist():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12398")
    dist.init_process_group(backend="nccl", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    patch_target_module(
        "transformers.loss.loss_utils.fixed_cross_entropy",
        hf_fixed_cross_entropy_none_reduction,
    )


def build_model(model_path, osft_rank_ratio=0.25):
    model = setup_model(
        model_name_or_path=model_path,
        use_liger_kernels=False,
        osft=True,
        osft_rank_ratio=osft_rank_ratio,
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
    )
    model, optimizer, lr_scheduler = setup_training_components(
        model,
        learning_rate=1e-3,
        num_warmup_steps=0,
        lr_scheduler="constant",
        compile_model=False,
    )
    return model, optimizer, lr_scheduler


def train_step(model, optimizer, lr_scheduler, input_ids, labels):
    optimizer.zero_grad()
    output = model(input_ids=input_ids, labels=labels)
    loss = output.loss.float().sum() / input_ids.shape[0]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    lr_scheduler.step()
    return loss.item()


def measure_with_cuda_events(model, optimizer, lr_scheduler, input_ids, labels, num_steps=20):
    """Measure step time with CUDA events (no profiler overhead)."""
    step_times = []
    for _ in range(num_steps):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        train_step(model, optimizer, lr_scheduler, input_ids, labels)
        end.record()
        torch.cuda.synchronize()
        step_times.append(start.elapsed_time(end))
    return step_times


def measure_with_profiler(model, optimizer, lr_scheduler, input_ids, labels, num_steps=10):
    """Measure with torch.profiler to get record_function breakdown."""
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(num_steps):
            with record_function("osft::training_step"):
                train_step(model, optimizer, lr_scheduler, input_ids, labels)

    return prof


def extract_times(prof, num_steps):
    """Extract device times for OSFT annotations from profiler.

    Times from torch.profiler are in microseconds. We convert to ms
    and average per step.
    """
    avgs = prof.key_averages()

    result = {}
    for evt in avgs:
        if evt.key.startswith("osft::"):
            result[evt.key] = {
                "device_time_ms": evt.device_time_total / 1000.0 / num_steps,
                "cpu_time_ms": evt.cpu_time_total / 1000.0 / num_steps,
                "count": evt.count // num_steps,
            }
    return result


def main():
    parser = argparse.ArgumentParser(description="OSFT projection overhead profiler")
    parser.add_argument("--small", action="store_true", help="Use tiny model (hidden=64, 2 layers)")
    parser.add_argument("--hidden-size", type=int, default=None, help="Override hidden_size")
    parser.add_argument("--num-layers", type=int, default=None, help="Override num_hidden_layers")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--measure-steps", type=int, default=10, help="Steps for CUDA event measurement")
    parser.add_argument("--profile-steps", type=int, default=5, help="Steps for profiler measurement")
    parser.add_argument("--urr", type=float, default=0.25, help="OSFT unfreeze rank ratio")
    args = parser.parse_args()

    import tempfile

    torch.manual_seed(42)
    setup_dist()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    tmpdir = os.path.join(tempfile.gettempdir(), "bench_osft_projection_model")

    config_overrides = {}
    if args.small:
        config_overrides = dict(SMALL_CONFIG)
    if args.hidden_size:
        config_overrides["hidden_size"] = args.hidden_size
    if args.num_layers:
        config_overrides["num_hidden_layers"] = args.num_layers

    effective_cfg = dict(LLAMA_8B_SHAPES_CONFIG)
    effective_cfg.update(config_overrides)

    if rank == 0:
        desc = f"hidden={effective_cfg['hidden_size']}, {effective_cfg['num_hidden_layers']} layers"
        print(f"Creating model: {desc}")
        if os.path.exists(os.path.join(tmpdir, "config.json")):
            print(f"  Reusing cached model at {tmpdir}")
        else:
            os.makedirs(tmpdir, exist_ok=True)
            create_model(tmpdir, config_overrides if config_overrides else None)
    cfg = effective_cfg

    dist.barrier()
    model_path = tmpdir

    try:
        model, optimizer, lr_scheduler = build_model(model_path, osft_rank_ratio=args.urr)

        vocab_size = cfg["vocab_size"]
        input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=f"cuda:{local_rank}")
        labels = input_ids.clone()

        # Warmup
        if rank == 0:
            print(f"Warming up ({args.warmup_steps} steps)...")
        for _ in range(args.warmup_steps):
            train_step(model, optimizer, lr_scheduler, input_ids, labels)
        torch.cuda.synchronize()

        # Phase 1: CUDA events for clean wall-clock step time
        if rank == 0:
            print(f"Measuring step time with CUDA events ({args.measure_steps} steps)...")
        step_times = measure_with_cuda_events(
            model, optimizer, lr_scheduler, input_ids, labels, num_steps=args.measure_steps
        )

        # Phase 2: Profiler for record_function breakdown
        if rank == 0:
            print(f"Profiling with torch.profiler ({args.profile_steps} steps)...")
        prof = measure_with_profiler(
            model, optimizer, lr_scheduler, input_ids, labels, num_steps=args.profile_steps
        )
        times = extract_times(prof, args.profile_steps)

        if rank == 0:
            gpu_name = torch.cuda.get_device_name(local_rank)
            mode = f"{world_size}-GPU distributed" if world_size > 1 else "single-GPU"

            median_step = sorted(step_times)[len(step_times) // 2]
            mean_step = sum(step_times) / len(step_times)
            min_step = min(step_times)
            max_step = max(step_times)

            effective_cfg = dict(LLAMA_8B_SHAPES_CONFIG)
            effective_cfg.update(config_overrides)
            desc = f"hidden={effective_cfg['hidden_size']}, {effective_cfg['num_hidden_layers']} layers"

            print()
            print(f"OSFT Projection Profiling — {world_size}x {gpu_name}, {mode}")
            print(f"Model: Llama ({desc}), OSFT URR={args.urr}")
            print(f"Batch: {args.batch_size} x {args.seq_len} tokens")
            print("=" * 70)

            print(f"\n  Step time (CUDA events, {args.measure_steps} steps):")
            print(f"    Median: {median_step:.3f} ms")
            print(f"    Mean:   {mean_step:.3f} ms")
            print(f"    Min:    {min_step:.3f} ms")
            print(f"    Max:    {max_step:.3f} ms")

            print(f"\n  Profiler breakdown (per step, device time):")

            grad_dev = times.get("osft::project_gradients", {}).get("device_time_ms", 0)
            param_dev = times.get("osft::project_parameters", {}).get("device_time_ms", 0)
            step_dev = times.get("osft::training_step", {}).get("device_time_ms", 0)
            total_proj = grad_dev + param_dev

            if step_dev > 0:
                print(f"    osft::training_step:       {step_dev:8.3f} ms")
                print(f"    osft::project_gradients:   {grad_dev:8.3f} ms  ({grad_dev / step_dev * 100:5.1f}%)")
                print(f"    osft::project_parameters:  {param_dev:8.3f} ms  ({param_dev / step_dev * 100:5.1f}%)")
                print(f"    Total projection overhead: {total_proj:8.3f} ms  ({total_proj / step_dev * 100:5.1f}%)")
                print(f"    Forward + backward (est):  {step_dev - total_proj:8.3f} ms  ({(step_dev - total_proj) / step_dev * 100:5.1f}%)")
            else:
                print(f"    osft::project_gradients:   {grad_dev:8.3f} ms")
                print(f"    osft::project_parameters:  {param_dev:8.3f} ms")
                print(f"    Total projection overhead: {total_proj:8.3f} ms")
                print(f"    (training_step device time = 0, using CPU times instead)")
                grad_cpu = times.get("osft::project_gradients", {}).get("cpu_time_ms", 0)
                param_cpu = times.get("osft::project_parameters", {}).get("cpu_time_ms", 0)
                step_cpu = times.get("osft::training_step", {}).get("cpu_time_ms", 0)
                total_proj_cpu = grad_cpu + param_cpu
                if step_cpu > 0:
                    print(f"\n  CPU time breakdown (per step):")
                    print(f"    osft::training_step:       {step_cpu:8.3f} ms")
                    print(f"    osft::project_gradients:   {grad_cpu:8.3f} ms  ({grad_cpu / step_cpu * 100:5.1f}%)")
                    print(f"    osft::project_parameters:  {param_cpu:8.3f} ms  ({param_cpu / step_cpu * 100:5.1f}%)")
                    print(f"    Total projection overhead: {total_proj_cpu:8.3f} ms  ({total_proj_cpu / step_cpu * 100:5.1f}%)")

            # Sub-breakdown for distributed path
            if world_size > 1:
                print(f"\n  Distributed projection sub-breakdown (per step):")
                for key in ["osft::U_coeff_collect", "osft::U_allreduce", "osft::U_apply", "osft::V_projection"]:
                    info = times.get(key, {})
                    dev_ms = info.get("device_time_ms", 0)
                    cpu_ms = info.get("cpu_time_ms", 0)
                    count = info.get("count", 0)
                    print(f"    {key:30s}  dev={dev_ms:8.3f} ms  cpu={cpu_ms:8.3f} ms  (count={count})")

            # Also show the full profiler table for reference
            print(f"\n  Top 30 CUDA ops (profiler):")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
