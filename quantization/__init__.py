"""Post-training quantization of Hugging Face causal LMs with torchao.

Package layout
--------------
- ``schemes``   : registry of supported torchao quantization schemes.
- ``hardware``  : GPU capability checks so unsupported schemes fail early.
- ``quantizer`` : load -> quantize -> save a Hugging Face checkpoint that vLLM can serve.
- ``report``    : size / latency / sample-generation sanity report.

Typical use (see ``run_quantize.py`` and ``docs/quantization.md``)::

    from quantization import QuantizationJob, quantize_and_save

    job = QuantizationJob(
        model_path="~/models/Qwen2.5-1.5B-Instruct",
        output_dir="~/models/quantized/qwen2.5-1.5b-instruct-int8_weight_only",
        scheme="int8_weight_only",
    )
    quantize_and_save(job)
"""

from quantization.schemes import QuantOptions, QuantScheme, get_scheme, list_schemes, register_scheme
from quantization.quantizer import QuantizationJob, quantize_and_save

__all__ = [
    "QuantOptions",
    "QuantScheme",
    "QuantizationJob",
    "get_scheme",
    "list_schemes",
    "quantize_and_save",
    "register_scheme",
]
