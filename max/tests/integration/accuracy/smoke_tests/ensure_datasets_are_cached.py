#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

"""Prepare the datasets used by serve smoke tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

DATASETS = (
    ("openai/gsm8k", "main"),
    ("HuggingFaceM4/ChartQA", None),
)


def load_datasets(*, quiet: bool = False) -> None:
    from datasets import DownloadMode, load_dataset

    for path, name in DATASETS:
        if not quiet:
            print(f"Preparing {path}" + (f" ({name})" if name else ""))
        args = (path, name) if name else (path,)
        load_dataset(*args, download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)


@contextmanager
def huggingface_offline() -> Iterator[None]:
    import datasets.config
    import huggingface_hub.constants

    previous_datasets_offline = datasets.config.HF_DATASETS_OFFLINE
    previous_hub_offline = huggingface_hub.constants.HF_HUB_OFFLINE
    datasets.config.HF_DATASETS_OFFLINE = True
    huggingface_hub.constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        datasets.config.HF_DATASETS_OFFLINE = previous_datasets_offline
        huggingface_hub.constants.HF_HUB_OFFLINE = previous_hub_offline


def datasets_are_cached() -> bool:
    try:
        with huggingface_offline():
            load_datasets(quiet=True)
    except Exception:
        return False
    return True


def main() -> None:
    if datasets_are_cached():
        print("Eval datasets already cached")
        return

    print("Downloading/preparing eval datasets")
    load_datasets()


if __name__ == "__main__":
    main()
