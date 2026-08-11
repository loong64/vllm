# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.cpu_resource_utils import LogicalCPUInfo
from vllm.utils.ompmultiprocessing import OMPProcessManager


def test_loongarch_auto_binding_uses_all_logical_cpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = object.__new__(OMPProcessManager)
    manager.reserve_cpu_num = 1
    cpus = [LogicalCPUInfo(id=0), LogicalCPUInfo(id=1)]
    selected_cpus: list[LogicalCPUInfo] = []

    monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", "auto")
    monkeypatch.setattr(
        current_platform,
        "get_cpu_architecture",
        lambda: CpuArchEnum.LOONGARCH,
    )

    def get_autobind_cpu_ids(selector):
        selected_cpus.extend(selector(cpus))
        return [cpus], []

    manager._get_autobind_cpu_ids = get_autobind_cpu_ids
    manager._parse_omp_threads_bind_env()

    assert selected_cpus == cpus
    assert manager.cpu_lists == [[0, 1]]
