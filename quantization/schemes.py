"""Registry of torchao quantization schemes.

Each scheme is a small, declarative ``QuantScheme`` that knows:

* how to build the torchao ``AOBaseConfig`` that describes the quantization, and
* which hardware it needs (CUDA, minimum compute capability).

New schemes are added by decorating a builder function with ``@register_scheme``;
nothing else in the package needs to change.

torchao is imported lazily inside the builders so that listing schemes (e.g.
``run_quantize.py --list``) works even where torchao is not installed.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QuantOptions:
    """User-tunable knobs shared by all schemes.

    Attributes:
        group_size: Number of weights sharing one scale (group-wise schemes only,
            currently INT4). Smaller = more accurate but larger checkpoint.
            Typical values: 32, 64, 128, 256.
        int4_packing_format: torchao packing layout for INT4 weights.
            ``"tile_packed_to_4d"`` uses the tinygemm kernel, which runs on Ampere
            without extra dependencies. Ignored by torchao versions without this
            option.
        extra: Free-form keyword arguments forwarded verbatim to the torchao
            config constructor (advanced use).
    """

    group_size: int = 128
    int4_packing_format: str = "tile_packed_to_4d"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuantScheme:
    """A named torchao quantization recipe.

    Attributes:
        name: Identifier used in ``config.py`` and on the CLI.
        description: One-line human-readable summary.
        build: Callable turning ``QuantOptions`` into a torchao ``AOBaseConfig``.
        requires_cuda: Whether a CUDA GPU is mandatory.
        min_compute_capability: Minimum ``(major, minor)`` CUDA capability, or
            ``None`` if any GPU (or CPU, when ``requires_cuda`` is False) works.
    """

    name: str
    description: str
    build: Callable[[QuantOptions], Any]
    requires_cuda: bool = True
    min_compute_capability: tuple[int, int] | None = None


_REGISTRY: dict[str, QuantScheme] = {}


def register_scheme(
    name: str,
    description: str,
    *,
    requires_cuda: bool = True,
    min_compute_capability: tuple[int, int] | None = None,
) -> Callable[[Callable[[QuantOptions], Any]], Callable[[QuantOptions], Any]]:
    """Decorator that registers a torchao config builder under ``name``."""

    def decorator(builder: Callable[[QuantOptions], Any]) -> Callable[[QuantOptions], Any]:
        if name in _REGISTRY:
            raise ValueError(f"Quantization scheme '{name}' is already registered.")
        _REGISTRY[name] = QuantScheme(
            name=name,
            description=description,
            build=builder,
            requires_cuda=requires_cuda,
            min_compute_capability=min_compute_capability,
        )
        return builder

    return decorator


def get_scheme(name: str) -> QuantScheme:
    """Return the registered scheme called ``name`` or raise a helpful error."""
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown quantization scheme '{name}'. Available: {available}") from None


def list_schemes() -> list[QuantScheme]:
    """Return all registered schemes, sorted by name."""
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _supported_kwargs(cls: type, candidates: dict[str, Any]) -> dict[str, Any]:
    """Keep only the kwargs that ``cls.__init__`` accepts.

    torchao's config signatures evolve between releases; filtering lets the same
    code run on several versions instead of crashing on an unknown argument.
    """
    params = inspect.signature(cls).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(candidates)
    return {k: v for k, v in candidates.items() if k in params}


# --------------------------------------------------------------------------- #
# Built-in schemes
# --------------------------------------------------------------------------- #


@register_scheme(
    "int8_weight_only",
    "INT8 weights, BF16 activations (W8A16). ~2x smaller, near-lossless. Safest default.",
    requires_cuda=False,
)
def _int8_weight_only(opts: QuantOptions) -> Any:
    from torchao.quantization import Int8WeightOnlyConfig

    return Int8WeightOnlyConfig(**opts.extra)


@register_scheme(
    "int4_weight_only",
    "Group-wise INT4 weights, BF16 activations (W4A16). ~4x smaller, small accuracy loss.",
    min_compute_capability=(8, 0),
)
def _int4_weight_only(opts: QuantOptions) -> Any:
    from torchao.quantization import Int4WeightOnlyConfig

    kwargs = _supported_kwargs(
        Int4WeightOnlyConfig,
        {"group_size": opts.group_size, "int4_packing_format": opts.int4_packing_format},
    )
    kwargs.update(opts.extra)
    return Int4WeightOnlyConfig(**kwargs)


@register_scheme(
    "int8_dynamic_act_int8_weight",
    "INT8 weights + dynamically quantized INT8 activations (W8A8). Faster for large batches.",
    min_compute_capability=(7, 5),
)
def _int8_dynamic(opts: QuantOptions) -> Any:
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig

    return Int8DynamicActivationInt8WeightConfig(**opts.extra)


@register_scheme(
    "float8_dynamic_act_float8_weight",
    "FP8 (e4m3) weights + dynamic FP8 activations, per-row scales. Needs Ada/Hopper (sm89+).",
    min_compute_capability=(8, 9),
)
def _float8_dynamic(opts: QuantOptions) -> Any:
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow

    kwargs: dict[str, Any] = {"granularity": PerRow()}
    kwargs.update(opts.extra)
    return Float8DynamicActivationFloat8WeightConfig(**kwargs)
