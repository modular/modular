# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import functools
import os
import pickle
import shutil
import subprocess
from collections.abc import Iterable
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import rich

LINE: str = "\n" + 70 * "-"


def _percentage(x: int, y: int) -> int:
    if x > 0 and y > 0:
        return int((x / y) * 100.0)
    return 0


def pretty_exception_handler(exception_type, exception, traceback) -> None:  # noqa: ANN001
    rich.print(f"[bold red]{exception_type.__name__}[/bold red]: {exception}")


def store_pickle(path: Path | str, data: Any) -> None:
    """Serialize data to a pickle file."""
    with Path(path).open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path | str) -> Any:
    """Deserialize data from a pickle file."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def flatten(value: int | object | Iterable) -> list[Any]:
    """Flatten nested iterables into a single list."""
    if not isinstance(value, Iterable) or isinstance(value, str):
        return [value]
    return [
        item
        for sublist in (flatten(item) for item in value)
        for item in sublist
    ]


def _get_core_count() -> int:
    try:
        # The 'os.sched_getaffinity' method is only available on some Unix platforms
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined, unused-ignore]
    except AttributeError:
        # To cover other platforms, including mac
        return cpu_count()


def _get_visible_device_prefix(target_accelerator: str = "") -> str:
    """Returns the environment variable prefix for visible devices based on accelerator type."""
    if "nvidia" in target_accelerator or "cuda" in target_accelerator:
        return "CUDA_VISIBLE_DEVICES"
    elif "amd" in target_accelerator:
        return "ROCR_VISIBLE_DEVICES"
    else:
        return ""


@functools.cache
def get_nvidia_smi():  # noqa: ANN201
    return shutil.which("nvidia-smi")


def check_gpu_clock() -> None:
    nvidia_smi = get_nvidia_smi()
    if not nvidia_smi:
        return
    output = subprocess.check_output(
        [
            nvidia_smi,
            "--query-gpu",
            "persistence_mode",
            "--format",
            "csv",
        ],
    )

    # We check for persistence here as a proxy to check if setup-gpu-clock.sh
    # has been run. This is not exact, but should cover most cases. Checking for
    # the clock frequency is more complicated since the frequencies changes per
    # GPU.
    if "Disabled" in output.decode("utf-8"):
        raise Exception(
            "the clock frequency for the GPU is not locked, please run"
            " `sudo utils/setup-gpu-clock.sh` to ensure the frequencies"
            " and power of the GPU are locked to get consistent"
            " benchmarking behavior."
        )


target_accelerator_values = {
    "NVIDIA": [
        "nvidia:sm_52",
        "nvidia:sm_60",
        "nvidia:sm_61",
        "nvidia:sm_75",
        "nvidia:sm_80",
        "nvidia:sm_86",
        "nvidia:sm_87",
        "nvidia:sm_89",
        "nvidia:sm_90",
        "nvidia:sm_90a",
        "nvidia:sm_100",
        "nvidia:sm_100a",
        "nvidia:sm_120",
        "nvidia:sm_120a",
    ],
    "AMD": [
        "amdgpu:mi300x",
        "amdgpu:mi355x",
        "amdgpu:gfx942",
        "amdgpu:gfx950",
        "amdgpu:gfx1030",
        "amdgpu:gfx1100",
        "amdgpu:gfx1101",
        "amdgpu:gfx1102",
        "amdgpu:gfx1103",
        "amdgpu:gfx1150",
        "amdgpu:gfx1151",
        "amdgpu:gfx1152",
        "amdgpu:gfx1200",
        "amdgpu:gfx1201",
    ],
    "Apple": ["metal:1", "metal:2", "metal:3", "metal:4"],
}


@functools.cache
def get_target_accelerator_helpstr() -> str:
    helpstr = ""
    for arch, target_list in target_accelerator_values.items():
        helpstr += f"\n\n# {arch}\n\n"
        helpstr += ",\n".join([f"'{x}'" for x in target_list])
    return helpstr


def check_valid_target_accelerator(target_accelerator: str) -> bool:
    return target_accelerator in flatten(target_accelerator_values.values())
