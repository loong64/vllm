# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import platform

import pytest

from vllm.platforms.interface import CpuArchEnum, Platform


@pytest.mark.parametrize(
    ("machine", "expected_architecture"),
    [
        ("loongarch64", CpuArchEnum.LOONGARCH),
        ("LoongArch64", CpuArchEnum.LOONGARCH),
    ],
)
def test_get_cpu_architecture_recognizes_loongarch(
    monkeypatch: pytest.MonkeyPatch,
    machine: str,
    expected_architecture: CpuArchEnum,
) -> None:
    monkeypatch.setattr(platform, "machine", lambda: machine)

    assert Platform.get_cpu_architecture() == expected_architecture
