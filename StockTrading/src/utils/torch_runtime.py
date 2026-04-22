from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import chain
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class TorchRuntime:
    requested_device: str
    device: torch.device
    accelerator_available: bool
    backend: str
    device_name: str
    hip_version: Optional[str]
    torch_version: str

    @property
    def using_gpu(self) -> bool:
        return self.device.type != 'cpu'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'requested_device': self.requested_device,
            'resolved_device': str(self.device),
            'accelerator_available': self.accelerator_available,
            'backend': self.backend,
            'device_name': self.device_name,
            'hip_version': self.hip_version,
            'torch_version': self.torch_version,
        }


def resolve_torch_runtime(
    preferred_device: str | torch.device = 'auto',
    require_accelerator: bool = False,
) -> TorchRuntime:
    requested_device = str(preferred_device or 'auto').lower()

    if isinstance(preferred_device, torch.device):
        resolved_device = preferred_device
    elif requested_device.startswith('cpu'):
        resolved_device = torch.device('cpu')
    elif requested_device in {'auto', 'gpu', 'cuda', 'rocm', 'hip'} or requested_device.startswith('cuda'):
        if torch.cuda.is_available():
            resolved_device = torch.device(requested_device if requested_device.startswith('cuda:') else 'cuda')
        elif require_accelerator:
            raise RuntimeError('GPU acceleration was requested but PyTorch could not access an accelerator.')
        else:
            resolved_device = torch.device('cpu')
    else:
        resolved_device = torch.device(requested_device)
        if resolved_device.type == 'cuda' and not torch.cuda.is_available():
            if require_accelerator:
                raise RuntimeError('GPU acceleration was requested but PyTorch could not access an accelerator.')
            resolved_device = torch.device('cpu')

    hip_version = getattr(torch.version, 'hip', None)
    accelerator_available = torch.cuda.is_available()

    if resolved_device.type == 'cuda' and accelerator_available:
        device_name = torch.cuda.get_device_name(resolved_device)
        backend = 'rocm' if hip_version else 'cuda'
    else:
        device_name = 'CPU'
        backend = 'cpu'

    return TorchRuntime(
        requested_device=requested_device,
        device=resolved_device,
        accelerator_available=accelerator_available,
        backend=backend,
        device_name=device_name,
        hip_version=hip_version,
        torch_version=torch.__version__,
    )


def print_torch_runtime_summary(runtime: TorchRuntime) -> None:
    print(f"\n{'=' * 60}")
    print('PyTorch Runtime Summary')
    print('=' * 60)
    print(f"Requested device: {runtime.requested_device}")
    print(f"Resolved device:  {runtime.device}")
    print(f"Backend:          {runtime.backend}")
    print(f"Device name:      {runtime.device_name}")
    print(f"Torch version:    {runtime.torch_version}")
    if runtime.hip_version:
        print(f"HIP version:      {runtime.hip_version}")
    print('=' * 60 + '\n')


def configure_global_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = False,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(deterministic, warn_only=True)

    for backend_name in ('cudnn', 'miopen'):
        backend = getattr(torch.backends, backend_name, None)
        if backend is None:
            continue
        if hasattr(backend, 'deterministic'):
            backend.deterministic = deterministic
        if hasattr(backend, 'benchmark'):
            backend.benchmark = benchmark if not deterministic else False


def get_model_device(model: nn.Module) -> torch.device:
    for tensor in chain(model.parameters(), model.buffers()):
        return tensor.device
    return torch.device('cpu')


def create_grad_scaler(device: torch.device, enabled: bool = False):
    active = enabled and device.type != 'cpu'
    amp_module = getattr(torch, 'amp', None)

    if amp_module is not None and hasattr(amp_module, 'GradScaler'):
        try:
            return amp_module.GradScaler(device.type, enabled=active)
        except TypeError:
            return amp_module.GradScaler(enabled=active)

    if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'GradScaler'):
        return torch.cuda.amp.GradScaler(enabled=active)

    raise RuntimeError('Automatic mixed precision is not available in this PyTorch build.')


def autocast_context(device: torch.device, enabled: bool = False):
    active = enabled and device.type != 'cpu'
    amp_module = getattr(torch, 'amp', None)

    if amp_module is not None and hasattr(amp_module, 'autocast'):
        try:
            return amp_module.autocast(device_type=device.type, enabled=active)
        except TypeError:
            pass

    if device.type == 'cuda' and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        return torch.cuda.amp.autocast(enabled=active)

    return nullcontext()


def synchronize_device(device: torch.device) -> None:
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def get_memory_stats(device: torch.device) -> Dict[str, float]:
    if device.type != 'cuda' or not torch.cuda.is_available():
        return {
            'memory_allocated_mb': 0.0,
            'memory_reserved_mb': 0.0,
            'max_memory_allocated_mb': 0.0,
            'max_memory_reserved_mb': 0.0,
        }

    scale = 1024 ** 2
    return {
        'memory_allocated_mb': torch.cuda.memory_allocated(device) / scale,
        'memory_reserved_mb': torch.cuda.memory_reserved(device) / scale,
        'max_memory_allocated_mb': torch.cuda.max_memory_allocated(device) / scale,
        'max_memory_reserved_mb': torch.cuda.max_memory_reserved(device) / scale,
    }
