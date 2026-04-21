import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architectures.lstm_clean import get_model
from src.utils.torch_runtime import (
    autocast_context,
    create_grad_scaler,
    get_memory_stats,
    print_torch_runtime_summary,
    reset_peak_memory_stats,
    resolve_torch_runtime,
    synchronize_device,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Benchmark PyTorch LSTM GPU runtime')
    parser.add_argument('--model-type', type=str, default='enhanced', choices=['enhanced', 'simple'])
    parser.add_argument('--input-dim', type=int, default=12)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--output-dim', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--sequence-length', type=int, default=60)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--benchmark-steps', type=int, default=50)
    parser.add_argument('--transfer-steps', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--mixed-precision', action='store_true')
    return parser.parse_args()


def percentile(values, fraction):
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def measure_transfer_time(cpu_tensor, runtime, steps):
    if runtime.device.type == 'cpu':
        return 0.0

    durations = []
    for _ in range(steps):
        synchronize_device(runtime.device)
        start = time.perf_counter()
        gpu_tensor = cpu_tensor.to(runtime.device, non_blocking=True)
        synchronize_device(runtime.device)
        durations.append(time.perf_counter() - start)
        del gpu_tensor
    return statistics.mean(durations) * 1000


def run_train_step(model, criterion, optimizer, inputs, targets, runtime, scaler):
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(runtime.device, enabled=scaler.is_enabled()):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return loss.item()


def main():
    args = parse_arguments()
    runtime = resolve_torch_runtime(args.device)
    print_torch_runtime_summary(runtime)

    model = get_model(
        model_type=args.model_type,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_dim=args.output_dim,
    ).to(runtime.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = create_grad_scaler(
        runtime.device,
        enabled=args.mixed_precision and runtime.device.type != 'cpu',
    )

    inputs = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.input_dim,
        device=runtime.device,
    )
    targets = torch.randn(args.batch_size, args.output_dim, device=runtime.device)

    host_batch = torch.randn(args.batch_size, args.sequence_length, args.input_dim)
    if runtime.device.type != 'cpu':
        host_batch = host_batch.pin_memory()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    transfer_ms = measure_transfer_time(host_batch, runtime, args.transfer_steps)

    if runtime.device.type != 'cpu':
        reset_peak_memory_stats(runtime.device)

    for _ in range(args.warmup_steps):
        run_train_step(model, criterion, optimizer, inputs, targets, runtime, scaler)

    step_times = []
    losses = []
    for _ in range(args.benchmark_steps):
        synchronize_device(runtime.device)
        start = time.perf_counter()
        loss_value = run_train_step(model, criterion, optimizer, inputs, targets, runtime, scaler)
        synchronize_device(runtime.device)
        step_times.append(time.perf_counter() - start)
        losses.append(loss_value)

    step_times_ms = [duration * 1000 for duration in step_times]
    total_samples = args.batch_size * args.benchmark_steps
    total_time = sum(step_times)
    samples_per_second = total_samples / total_time if total_time > 0 else 0.0
    memory_stats = get_memory_stats(runtime.device)

    print('=' * 60)
    print('Benchmark Results')
    print('=' * 60)
    print(f'Model type:             {args.model_type}')
    print(f'Parameter count:        {parameter_count:,}')
    print(f'Batch size:             {args.batch_size}')
    print(f'Sequence length:        {args.sequence_length}')
    print(f'Mixed precision:        {scaler.is_enabled()}')
    print(f'Host->device mean ms:   {transfer_ms:.3f}')
    print(f'Train step mean ms:     {statistics.mean(step_times_ms):.3f}')
    print(f'Train step median ms:   {statistics.median(step_times_ms):.3f}')
    print(f'Train step p90 ms:      {percentile(step_times_ms, 0.90):.3f}')
    print(f'Samples per second:     {samples_per_second:.2f}')
    print(f'Final loss:             {losses[-1]:.6f}')
    print(f"Memory allocated MB:    {memory_stats['memory_allocated_mb']:.2f}")
    print(f"Memory reserved MB:     {memory_stats['memory_reserved_mb']:.2f}")
    print(f"Peak allocated MB:      {memory_stats['max_memory_allocated_mb']:.2f}")
    print(f"Peak reserved MB:       {memory_stats['max_memory_reserved_mb']:.2f}")
    print('=' * 60)


if __name__ == '__main__':
    main()
