"""Lightweight tests for the quantization package (no GPU / torchao required).

Run with:  python -m pytest tests/
"""

import pytest

from quantization.hardware import DeviceInfo, UnsupportedHardwareError, check_scheme_supported
from quantization.schemes import get_scheme, list_schemes

AMPERE = DeviceInfo(device="cuda", name="NVIDIA A100", compute_capability=(8, 0))
HOPPER = DeviceInfo(device="cuda", name="NVIDIA H100", compute_capability=(9, 0))
CPU = DeviceInfo(device="cpu", name="CPU", compute_capability=None)


def test_all_expected_schemes_registered():
    names = {s.name for s in list_schemes()}
    assert names == {
        "int8_weight_only",
        "int4_weight_only",
        "int8_dynamic_act_int8_weight",
        "float8_dynamic_act_float8_weight",
    }


def test_unknown_scheme_lists_alternatives():
    with pytest.raises(ValueError, match="Available"):
        get_scheme("int3_magic")


@pytest.mark.parametrize(
    "name", ["int8_weight_only", "int4_weight_only", "int8_dynamic_act_int8_weight"]
)
def test_ampere_supports_int_schemes(name):
    check_scheme_supported(get_scheme(name), AMPERE)


def test_fp8_rejected_on_ampere_but_ok_on_hopper():
    fp8 = get_scheme("float8_dynamic_act_float8_weight")
    with pytest.raises(UnsupportedHardwareError, match="8.9"):
        check_scheme_supported(fp8, AMPERE)
    check_scheme_supported(fp8, HOPPER)


def test_cpu_only_allows_int8_weight_only():
    check_scheme_supported(get_scheme("int8_weight_only"), CPU)
    with pytest.raises(UnsupportedHardwareError, match="CUDA"):
        check_scheme_supported(get_scheme("int4_weight_only"), CPU)
