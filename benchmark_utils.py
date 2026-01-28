"""
Inference benchmarking utilities for measuring speed and throughput.
Compatible with PyTorch models.
"""
import time
import torch
import numpy as np


def benchmark_inference(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: tuple,
    batch_size: int = 128,
    n_warmup: int = 20,
    n_iters: int = 200,
    use_mixed_precision: bool = False
):
    """
    Benchmark inference speed and throughput.

    Args:
        model: PyTorch model to benchmark
        device: torch.device ('cuda' or 'cpu')
        input_shape: Tuple defining input dimensions (excluding batch size)
                     e.g., (param_dim,) for scalar models or (param_dim, seq_len) for sequence models
        batch_size: Batch size for inference
        n_warmup: Number of warmup iterations
        n_iters: Number of timing iterations
        use_mixed_precision: Use torch.cuda.amp for FP16 inference

    Returns:
        dict with keys: avg_latency_ms, throughput_samples_per_sec, total_time_ms
    """
    model = model.to(device)
    model.eval()

    # Create dummy input based on shape
    if len(input_shape) == 1:
        # Single input (e.g., flattened parameters)
        dummy_input = torch.randn(batch_size, *input_shape, device=device)
    elif len(input_shape) == 2:
        # Two inputs (e.g., parameters + voltage grid)
        dummy_input = (
            torch.randn(batch_size, input_shape[0], device=device),
            torch.randn(batch_size, input_shape[1], device=device)
        )
    else:
        raise ValueError(f"Unsupported input_shape: {input_shape}")

    # Warmup
    print(f"Running {n_warmup} warmup iterations...")
    with torch.no_grad():
        for _ in range(n_warmup):
            if use_mixed_precision and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    if isinstance(dummy_input, tuple):
                        _ = model(*dummy_input)
                    else:
                        _ = model(dummy_input)
            else:
                if isinstance(dummy_input, tuple):
                    _ = model(*dummy_input)
                else:
                    _ = model(dummy_input)

    # Synchronize GPU before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timing
    print(f"Running {n_iters} timing iterations...")
    if device.type == 'cuda':
        # Use CUDA events for accurate GPU timing
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        starter.record()
        with torch.no_grad():
            for _ in range(n_iters):
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        if isinstance(dummy_input, tuple):
                            _ = model(*dummy_input)
                        else:
                            _ = model(dummy_input)
                else:
                    if isinstance(dummy_input, tuple):
                        _ = model(*dummy_input)
                    else:
                        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        total_ms = starter.elapsed_time(ender)
    else:
        # CPU timing
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                if isinstance(dummy_input, tuple):
                    _ = model(*dummy_input)
                else:
                    _ = model(dummy_input)
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

    avg_latency_ms = total_ms / n_iters
    throughput = batch_size / (avg_latency_ms / 1000.0)

    results = {
        'device': str(device),
        'batch_size': batch_size,
        'total_time_ms': float(total_ms),
        'avg_latency_ms': float(avg_latency_ms),
        'throughput_samples_per_sec': float(throughput),
        'use_mixed_precision': use_mixed_precision
    }

    print(f"\n{'='*60}")
    print(f"Benchmark Results")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Mixed precision: {use_mixed_precision}")
    print(f"  → Total time for {n_iters} runs: {total_ms:.1f} ms")
    print(f"  → Average latency per forward: {avg_latency_ms:.3f} ms")
    print(f"  → Throughput: {throughput:.0f} samples/s")
    print(f"{'='*60}\n")

    return results


def benchmark_multiple_batch_sizes(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: tuple,
    batch_sizes: list[int] = None,
    n_warmup: int = 20,
    n_iters: int = 200
):
    """
    Benchmark inference across multiple batch sizes.

    Args:
        model: PyTorch model
        device: torch.device
        input_shape: Input dimensions (excluding batch)
        batch_sizes: List of batch sizes to test
        n_warmup: Warmup iterations
        n_iters: Timing iterations

    Returns:
        list of result dicts
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 128, 512]

    results = []

    for bs in batch_sizes:
        print(f"\nBenchmarking batch size: {bs}")
        try:
            result = benchmark_inference(
                model=model,
                device=device,
                input_shape=input_shape,
                batch_size=bs,
                n_warmup=n_warmup,
                n_iters=n_iters
            )
            results.append(result)
        except RuntimeError as e:
            print(f"  → Skipped (OOM or error): {e}")
            continue

    return results


def profile_memory_usage(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: tuple,
    batch_size: int = 128
):
    """
    Profile GPU memory usage during inference.

    Args:
        model: PyTorch model
        device: torch.device (must be 'cuda')
        input_shape: Input dimensions
        batch_size: Batch size

    Returns:
        dict with memory stats in MB
    """
    if device.type != 'cuda':
        print("Memory profiling only available on CUDA devices.")
        return {}

    model = model.to(device)
    model.eval()

    # Create dummy input
    if len(input_shape) == 1:
        dummy_input = torch.randn(batch_size, *input_shape, device=device)
    elif len(input_shape) == 2:
        dummy_input = (
            torch.randn(batch_size, input_shape[0], device=device),
            torch.randn(batch_size, input_shape[1], device=device)
        )
    else:
        raise ValueError(f"Unsupported input_shape: {input_shape}")

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # Measure baseline
    baseline_allocated = torch.cuda.memory_allocated(device) / 1024**2
    baseline_reserved = torch.cuda.memory_reserved(device) / 1024**2

    # Run inference
    with torch.no_grad():
        if isinstance(dummy_input, tuple):
            _ = model(*dummy_input)
        else:
            _ = model(dummy_input)

    # Measure peak
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    current_allocated = torch.cuda.memory_allocated(device) / 1024**2

    results = {
        'baseline_allocated_mb': float(baseline_allocated),
        'baseline_reserved_mb': float(baseline_reserved),
        'peak_allocated_mb': float(peak_allocated),
        'peak_reserved_mb': float(peak_reserved),
        'current_allocated_mb': float(current_allocated),
        'inference_memory_mb': float(peak_allocated - baseline_allocated)
    }

    print(f"\n{'='*60}")
    print(f"Memory Usage Profile")
    print(f"{'='*60}")
    print(f"Baseline allocated: {baseline_allocated:.2f} MB")
    print(f"Peak allocated: {peak_allocated:.2f} MB")
    print(f"Inference memory: {results['inference_memory_mb']:.2f} MB")
    print(f"{'='*60}\n")

    return results


def estimate_flops(
    model: torch.nn.Module,
    input_shape: tuple,
    batch_size: int = 1
):
    """
    Estimate FLOPs using thop library.

    Requires: pip install thop

    Args:
        model: PyTorch model
        input_shape: Input dimensions
        batch_size: Batch size for estimation

    Returns:
        dict with flops and params
    """
    try:
        from thop import profile
    except ImportError:
        print("thop not installed. Install with: pip install thop")
        return {}

    model = model.cpu()  # Profile on CPU for consistency
    model.eval()

    # Create dummy input
    if len(input_shape) == 1:
        dummy_input = torch.randn(batch_size, *input_shape)
    elif len(input_shape) == 2:
        dummy_input = (
            torch.randn(batch_size, input_shape[0]),
            torch.randn(batch_size, input_shape[1])
        )
    else:
        raise ValueError(f"Unsupported input_shape: {input_shape}")

    # Profile
    if isinstance(dummy_input, tuple):
        flops, params = profile(model, inputs=dummy_input)
    else:
        flops, params = profile(model, inputs=(dummy_input,))

    results = {
        'flops': int(flops),
        'flops_giga': float(flops / 1e9),
        'params': int(params),
        'params_mega': float(params / 1e6)
    }

    print(f"\n{'='*60}")
    print(f"FLOPs & Parameters")
    print(f"{'='*60}")
    print(f"FLOPs: {results['flops_giga']:.2f} G")
    print(f"Params: {results['params_mega']:.2f} M")
    print(f"{'='*60}\n")

    return results


def create_benchmark_report(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: tuple,
    batch_sizes: list[int] = None,
    save_path: str = None
):
    """
    Create comprehensive benchmark report.

    Args:
        model: PyTorch model
        device: torch.device
        input_shape: Input dimensions
        batch_sizes: List of batch sizes to test
        save_path: Path to save JSON report

    Returns:
        dict with all benchmark results
    """
    import json

    report = {
        'model_info': {
            'name': model.__class__.__name__,
            'device': str(device),
            'input_shape': input_shape,
        },
        'inference_benchmarks': [],
        'memory_profile': {},
        'flops_estimate': {}
    }

    # 1. Benchmark multiple batch sizes
    print("="*80)
    print("Running inference benchmarks...")
    print("="*80)
    batch_results = benchmark_multiple_batch_sizes(
        model, device, input_shape, batch_sizes
    )
    report['inference_benchmarks'] = batch_results

    # 2. Memory profiling (GPU only)
    if device.type == 'cuda':
        print("\n" + "="*80)
        print("Profiling memory usage...")
        print("="*80)
        mem_result = profile_memory_usage(model, device, input_shape, batch_size=128)
        report['memory_profile'] = mem_result

    # 3. FLOP estimation
    print("\n" + "="*80)
    print("Estimating FLOPs...")
    print("="*80)
    flop_result = estimate_flops(model, input_shape, batch_size=1)
    report['flops_estimate'] = flop_result

    # Save report
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Benchmark report saved to {save_path}")

    return report
