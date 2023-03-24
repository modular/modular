# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path
import subprocess
import tarfile
import pytest
import tempfile

from enum import Enum
from pathlib import Path

_tmp_download_dir: Path


class ModelFormat(str, Enum):
    tflite = "tflite"
    tf = "tensorflow-savedmodel"
    onnx = "onnx"


def modular_derived_path() -> Path:
    modular_derived = os.getenv("MODULAR_DERIVED_PATH")
    assert modular_derived is not None
    return Path(modular_derived)


def modular_base_path() -> Path:
    modular_derived = os.getenv("MODULAR_PATH")
    assert modular_derived is not None
    return Path(modular_derived)


def download_dest(format: ModelFormat) -> Path:
    return _tmp_download_dir / format.value


def fixture_generator(model_key: str, format: ModelFormat) -> Path:
    """Downloads the model pointed to by `model_key` in the desired `format` and returns the path to it"""
    subprocess.run(
        [
            "model",
            "download",
            "--format",
            format.value,
            modular_base_path() / "Models" / f"{model_key}.yaml",
            "--weights-dir",
            download_dest(format),
        ]
    )
    if format == ModelFormat.tf:
        downloaded_file_path = download_dest(format) / f"{model_key}.tar.gz"
        tar = tarfile.open(downloaded_file_path)
        # turns out we don't need to extractall(), open mkdirs the folder we need.
        tar.close()
        extracted_dir = download_dest(format) / model_key
        assert os.path.isdir(extracted_dir)
        return Path(extracted_dir)
    else:
        # constructing paths from formats this way is brittle
        # but this currently works for both .onnx and .tflite
        downloaded_file_path = (
            download_dest(format) / f"{model_key}.{format.value}"
        )
        assert os.path.isfile(downloaded_file_path)
        return downloaded_file_path


@pytest.fixture(scope="session", autouse=True)
def session_ctx():
    # Setup code before first test, create tempdir to download to
    temp_dir = tempfile.TemporaryDirectory()
    global _tmp_download_dir
    _tmp_download_dir = Path(temp_dir.name)
    yield
    # Clean-up code after last test, delete tempdir
    temp_dir.cleanup()
