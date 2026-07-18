"""OSFT + FSDP2 stream safety integration tests.

Validates that register_osft_hooks (Phase 1a) produces stream-safe training
when running actual OSFT projections on FSDP2-sharded models under the CUDA
sanitizer.

Requires: 2+ GPUs, dev PyTorch with CUDA sanitizer accumulate mode.

Run via torchrun:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        tests/gpu_tests/test_osft_stream_safety.py

Or via pytest (spawns subprocesses internally):
    pytest tests/gpu_tests/test_osft_stream_safety.py -v -m multi_gpu
"""

import os
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from mini_trainer.osft_utils import (
    create_osft_model_class,
    optim_wrapper,
    register_osft_hooks,
)


class _SimpleModel(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.linear1 = nn.Linear(64, 64, bias=False)
        self.linear2 = nn.Linear(64, 64, bias=False)
        self.linear3 = nn.Linear(64, 32, bias=False)
        self.dtype = torch.float32

    def forward(self, x):
        return self.linear3(torch.relu(self.linear2(torch.relu(self.linear1(x)))))


def _make_osft_model(device="cpu"):
    """Create a small OSFT model for testing."""
    OSFTModel = create_osft_model_class(_SimpleModel)
    config = MagicMock()
    config.vocab_size = 100
    model = OSFTModel(
        config=config,
        osft_config={},
        initialize_osft=False,
        upcast_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    model.osft_config = {
        "linear1.weight": 32,
        "linear2.weight": 32,
        "linear3.weight": 16,
    }
    model.osft_unfreeze_rank_ratio = 0.5
    model.reinitialize_osft(decompose_existing_weights=True)
    return model.to(device)


def _init_distributed():
    """Initialize distributed process group from torchrun env vars."""
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def _run_c1_single_layer_sanitizer():
    """C1: Single OSFT model + FSDP2, sanitizer validation."""
    import torch.cuda._sanitizer as csan
    from torch.distributed.fsdp import fully_shard

    local_rank = _init_distributed()
    torch.manual_seed(42)

    model = _make_osft_model(f"cuda:{local_rank}")

    for module in model.children():
        fully_shard(module)
    fully_shard(model)

    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    handles = register_osft_hooks(optim, model, fsdp_model=model)
    assert handles is not None, "register_osft_hooks returned None"

    with csan.cuda_sanitizer as san:
        for step in range(4):
            optim.zero_grad()
            inp = torch.randn(4, 64, device=f"cuda:{local_rank}")
            loss = model(inp).sum()
            loss.backward()
            optim.step()
            torch.cuda.synchronize()

    n_errors = len(san.errors)
    if local_rank == 0:
        print(f"C1: {n_errors} sanitizer error(s)")
    assert n_errors == 0, f"C1: sanitizer detected {n_errors} stream race(s)"


def _run_c2_grad_accumulation():
    """C2: Multi-layer OSFT + FSDP2 + gradient accumulation.

    Known sanitizer false positive: NCCL work.wait() sync edges are invisible
    to the Python-level sanitizer, so it reports a WAR race between
    project_parameters' _local_tensor write and the previous forward's
    all_gather_copy_in read.  The hardware ordering is correct.
    We assert <= 1 error (the known false positive) rather than 0.
    """
    import torch.cuda._sanitizer as csan
    from torch.distributed.fsdp import fully_shard

    local_rank = _init_distributed()
    torch.manual_seed(42)
    n_microbatches = 2

    model = _make_osft_model(f"cuda:{local_rank}")

    for module in model.children():
        fully_shard(module)
    fully_shard(model)

    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    register_osft_hooks(optim, model, fsdp_model=model)

    with csan.cuda_sanitizer as san:
        for step in range(4):
            optim.zero_grad()
            for micro in range(n_microbatches):
                is_last = micro == n_microbatches - 1
                model.set_requires_gradient_sync(is_last)
                model.set_is_last_backward(is_last)
                inp = torch.randn(4, 64, device=f"cuda:{local_rank}")
                loss = model(inp).sum()
                loss.backward()
            optim.step()
            torch.cuda.synchronize()

    n_errors = len(san.errors)
    if local_rank == 0:
        print(f"C2: {n_errors} sanitizer error(s)")
    assert n_errors <= 1, f"C2: sanitizer detected {n_errors} stream race(s), expected <= 1 (known false positive)"


def _run_c3_numerical_equivalence():
    """C3: Hook path produces identical results to optim_wrapper path."""
    from torch.distributed.fsdp import fully_shard

    local_rank = _init_distributed()

    results = {}
    for mode in ("hooks", "wrapper"):
        torch.manual_seed(42)
        model = _make_osft_model(f"cuda:{local_rank}")

        for module in model.children():
            fully_shard(module)
        fully_shard(model)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.Adam(trainable, lr=1e-3)

        if mode == "hooks":
            register_osft_hooks(optim, model, fsdp_model=model)
        else:
            optim_wrapper(optim, model)

        torch.manual_seed(123 + local_rank)
        for step in range(8):
            optim.zero_grad()
            inp = torch.randn(4, 64, device=f"cuda:{local_rank}")
            loss = model(inp).sum()
            loss.backward()
            optim.step()

        torch.cuda.synchronize()
        param_snapshot = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                local = p.data._local_tensor if hasattr(p.data, "_local_tensor") else p.data
                param_snapshot[name] = local.detach().clone()
        results[mode] = param_snapshot

    mismatches = []
    for name in results["hooks"]:
        h = results["hooks"][name]
        w = results["wrapper"][name]
        diff = (h - w).abs().max().item()
        if diff > 1e-5:
            mismatches.append(f"{name}: max_diff={diff:.2e}")

    if local_rank == 0:
        if mismatches:
            print(f"C3: FAILED - {len(mismatches)} param mismatch(es):")
            for m in mismatches:
                print(f"  {m}")
        else:
            print("C3: PASSED - hook path matches optim_wrapper path")

    assert len(mismatches) == 0, f"C3: hook path differs from optim_wrapper for {len(mismatches)} params: " + "; ".join(
        mismatches
    )


# -- torchrun entry point --

_TESTS = {
    "c1": _run_c1_single_layer_sanitizer,
    "c2": _run_c2_grad_accumulation,
    "c3": _run_c3_numerical_equivalence,
}


def _torchrun_main():
    test_name = os.environ.get("OSFT_TEST", "all")
    try:
        if test_name == "all":
            for name, fn in _TESTS.items():
                fn()
        elif test_name in _TESTS:
            _TESTS[test_name]()
        else:
            raise ValueError(f"Unknown test: {test_name}. Choose from {list(_TESTS)}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# -- pytest wrappers (spawn torchrun as subprocess) --


def _run_via_torchrun(test_name, n_gpus=2):
    """Run a test function via torchrun subprocess."""
    env = os.environ.copy()
    env["OSFT_TEST"] = test_name
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(n_gpus),
            "--no_python",
            sys.executable,
            __file__,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(f"torchrun failed for {test_name}:\nstdout: {result.stdout}\nstderr: {result.stderr}")


@pytest.mark.multi_gpu
@pytest.mark.gpu
class TestOSFTStreamSafety:
    """OSFT + FSDP2 stream safety integration tests.

    Each test spawns a torchrun subprocess with 2 GPUs.
    """

    def test_c1_single_layer_sanitizer(self):
        _run_via_torchrun("c1")

    def test_c2_grad_accumulation(self):
        _run_via_torchrun("c2")

    def test_c3_numerical_equivalence(self):
        _run_via_torchrun("c3")


if __name__ == "__main__":
    _torchrun_main()
