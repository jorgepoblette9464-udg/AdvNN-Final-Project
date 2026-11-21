"""Hardware capability checks for quantization schemes.

Quantizing a 1.5B model only takes a minute, but finding out *after* that the
kernels do not exist on the GPU (e.g. FP8 on an Ampere A100) wastes time on a
shared server. ``check_scheme_supported`` fails fast with a clear message.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantization.schemes import QuantScheme


class UnsupportedHardwareError(RuntimeError):
    """Raised when the current device cannot run the requested scheme."""


@dataclass(frozen=True)
class DeviceInfo:
    """Summary of the device that will run quantization."""

    device: str  # "cuda" or "cpu"
    name: str
    compute_capability: tuple[int, int] | None  # None on CPU


def detect_device() -> DeviceInfo:
    """Return information about the first visible CUDA device, or the CPU."""
    import torch

    if torch.cuda.is_available():
        return DeviceInfo(
            device="cuda",
            name=torch.cuda.get_device_name(0),
            compute_capability=torch.cuda.get_device_capability(0),
        )
    return DeviceInfo(device="cpu", name="CPU", compute_capability=None)


def check_scheme_supported(scheme: QuantScheme, info: DeviceInfo | None = None) -> DeviceInfo:
    """Raise ``UnsupportedHardwareError`` if ``scheme`` cannot run on this machine.

    Returns the detected ``DeviceInfo`` so callers can reuse it.
    """
    info = info or detect_device()

    if scheme.requires_cuda and info.device != "cuda":
        raise UnsupportedHardwareError(
            f"Scheme '{scheme.name}' requires a CUDA GPU, but none is visible "
            "(check CUDA_VISIBLE_DEVICES / config.GPU_ID)."
        )

    if scheme.min_compute_capability and info.compute_capability is not None:
        if info.compute_capability < scheme.min_compute_capability:
            have = ".".join(map(str, info.compute_capability))
            need = ".".join(map(str, scheme.min_compute_capability))
            raise UnsupportedHardwareError(
                f"Scheme '{scheme.name}' needs compute capability >= {need}, "
                f"but '{info.name}' has {have}. "
                "On Ampere GPUs use an INT8 or INT4 scheme instead."
            )

    return info
