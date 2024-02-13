# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# Usage:
# - For C package
#       python3 All/test/API/python/test_p0_models.py --package_path /path/to/local/package

import argparse
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set in buildAndDeployMAX.yaml
TEST_ENV_ARCH = os.getenv("TEST_ENV_ARCH", "")
TEST_ENV_OS = os.getenv("TEST_ENV_OS", "")

MODULAR_PATH = Path.cwd()
MODELS_DIR = MODULAR_PATH / "test-models"
YAML_PATH = MODULAR_PATH / "Models"

# Frameworks
TF = "tf"
PYTORCH = "torch"
ONNX = "onnx"


@dataclass
class ModelMeta:
    s3_uri: str
    input_names: Sequence[str]
    input_specs: Sequence[str]
    output_names: Sequence[str]
    framework: str
    saved_model_dir_name: Optional[str] = None


top_models: Mapping[str, ModelMeta] = {
    "bert-base-uncased-seqlen-128": ModelMeta(
        s3_uri="s3://modular-model-storage/Tensorflow/bert-base-uncased_dynamic_seqlen_savedmodel.tar.gz",
        saved_model_dir_name="bert-base-uncased_dynamic_seqlen",
        input_names=["attention_mask", "input_ids", "token_type_ids"],
        input_specs=["1x128xsi32", "1x128xsi32", "1x128xsi32"],
        output_names=["last_hidden_state", "pooler_output"],
        framework=TF,
    ),
    "dlrm-rm1-multihot": ModelMeta(
        s3_uri="s3://modular-model-storage/dlrm/dlrm-facebook-1.tar.gz",
        saved_model_dir_name="dlrm-facebook-1",
        input_names=["ls_i", "xt"],
        input_specs=["4096x8x100xsi32", "4096x256xf32"],
        output_names=["output_0"],
        framework=TF,
    ),
    "dlrm-rm1": ModelMeta(
        s3_uri="s3://modular-model-storage/dlrm/dlrm_rm1.tar.gz",
        saved_model_dir_name="dlrm-rm1",
        input_names=["ls_i", "xt"],
        input_specs=["16384x8x1xsi32", "16384x256xf32"],
        output_names=["output_0"],
        framework=TF,
    ),
    "dlrm-rm2": ModelMeta(
        s3_uri="s3://modular-model-storage/dlrm/dlrm_rm2.tar.gz",
        saved_model_dir_name="dlrm-rm2",
        input_names=["ls_i", "xt"],
        input_specs=["2048x40x1xsi32", "2048x256xf32"],
        output_names=["output_0"],
        framework=TF,
    ),
    "dlrm-rm3": ModelMeta(
        s3_uri="s3://modular-model-storage/dlrm/dlrm-facebook-3.tar.gz",
        saved_model_dir_name="dlrm-rm3",
        input_names=["ls_i", "xt"],
        input_specs=["1024x10x1xsi32", "1024x2560xf32"],
        output_names=["output_0"],
        framework=TF,
    ),
    "clip-vit-large-patch14": ModelMeta(
        s3_uri="s3://modular-model-storage/Tensorflow/clip_vit_large_patch14_savedmodel.tar.gz",
        saved_model_dir_name="clip_vit_large_patch14_savedmodel",
        input_names=["pixel_values", "input_ids", "attention_mask"],
        input_specs=["1x3x224x224xf32", "1x16xsi32", "1x16xsi32"],
        output_names=[
            "image_embeds",
            "logits_per_image",
            "logits_per_text",
            "text_embeds",
            "text_model_output_last_hidden_state",
            "text_model_output_pooler_output",
            "vision_model_output_last_hidden_state",
            "vision_model_output_pooler_output",
        ],
        framework=TF,
    ),
    "bert-base-uncased-onnx": ModelMeta(
        s3_uri="s3://modular-model-storage/ONNX/bert-base-uncased.onnx",
        input_names=["input_ids"],
        input_specs=["1x5xsi32"],
        output_names=[
            "last_hidden_state",
            "pooler_output",
        ],
        framework=ONNX,
    ),
    "bert-base-uncased-torch": ModelMeta(
        s3_uri="s3://modular-model-storage/TorchScript/bert-base-uncased_pytorch.torchscript",
        input_names=["input_ids", "attention_mask", "input"],
        input_specs=["1x128xsi32", "1x128xsi32", "1x128xsi32"],
        output_names=[
            "result0",
            "result1",
        ],
        framework=PYTORCH,
    ),
}


def download_s3_object(
    s3_src: str, local_dst: Path, dir_name: Optional[str] = None
) -> Path:
    """Downloads and extract .tar.gz object at `s3_src` to file at `local_dst`
    """

    subprocess.run(
        ["aws", "s3", "cp", f"{s3_src}", f"{local_dst}"],
        check=True,
        capture_output=True,
    )
    filename = s3_src.split("/")[-1]

    local_file = local_dst / filename
    # ONNX models are not saved as tar archives, so we shouldn't try to unzip them
    if not filename.endswith(".tar.gz"):
        return local_file

    with tarfile.open(local_file, "r") as tar:
        tar.extractall(local_dst)

    if dir_name:
        return local_dst / dir_name

    # Remove tar.gz extension
    suffix_removed = str(local_file).replace(".tar.gz", "")
    return Path(suffix_removed)


@pytest.fixture(autouse=True, scope="module")
def remove_modular_env_variables():
    for key in os.environ:
        if "MODULAR" in key:
            del os.environ[key]

    # TODO(https://github.com/modularml/modular/issues/17843)
    assert "MODULAR_DERIVED_PATH" not in os.environ


def setup_environment_for_c(
    package_library_path: Path, package_binary_path: Path
):
    if platform.system() == "Windows":
        os.environ["Path"] = f"{os.environ['Path']};{package_binary_path}"
    else:
        os.environ["PATH"] = f"{os.environ['PATH']}:{package_binary_path}"

    os.environ["MODULAR_FRAMEWORK_ROOT"] = str(package_library_path)


def c_package_test(package_path: Path):
    package_library_path = package_path / "lib"
    package_binary_path = package_path / "bin"
    modular_api_executor_binary = package_binary_path / "modular-api-executor"

    setup_environment_for_c(package_library_path, package_binary_path)

    with tempfile.TemporaryDirectory() as test_dir:
        test_dir_path = Path(test_dir)

        common_modular_api_executor_args = (
            "--model-inputs=zeros  --num-threads=0"
            " --num-runs=1 --result-output-style=compact".split()
        )
        for _, value in top_models.items():
            model_path = download_s3_object(
                value.s3_uri, test_dir_path, value.saved_model_dir_name
            )
            input_names_args = [
                f"--input-names={name}" for name in value.input_names
            ]
            output_names_args = [
                f"--output-names={name}" for name in value.output_names
            ]
            input_specs_args = [
                f"--input-shapes={spec}" for spec in value.input_specs
            ]
            cmd = [
                str(modular_api_executor_binary),
                *input_names_args,
                *input_specs_args,
                *output_names_args,
                *common_modular_api_executor_args,
                model_path,
            ]
            print(f"Running {subprocess.list2cmdline(cmd)}")
            success = 0
            failure = 0
            total = 1
            for i in range(total):
                try:
                    proc = subprocess.run(cmd, check=True, capture_output=True)
                    success += 1
                except KeyboardInterrupt:
                    exit(1)
                except subprocess.CalledProcessError as e:
                    print(e.stderr.decode("utf-8"))
                    failure += 1
            print(f"Success: {success}")
            print(f"Failure: {failure}")
            print(f"Success Rate: {100 * success / total }%")
            if failure > 0:
                sys.exit(1)


def _get_np_dtype(dtype):
    import numpy as np
    from max import engine

    mtype = engine.DType
    if dtype == mtype.bool:
        return np.bool_
    elif dtype == mtype.int8:
        return np.int8
    elif dtype == mtype.int16:
        return np.int16
    elif dtype == mtype.int32:
        return np.int32
    elif dtype == mtype.int64:
        return np.int64
    elif dtype == mtype.uint8:
        return np.uint8
    elif dtype == mtype.uint16:
        return np.uint16
    elif dtype == mtype.uint32:
        return np.uint32
    elif dtype == mtype.uint64:
        return np.uint64
    elif dtype == mtype.float16:
        return np.float16
    elif dtype == mtype.float32:
        return np.float32
    elif dtype == mtype.float64:
        return np.float64


def _generate_test_inputs(model):
    import numpy as np

    _batch_size = 1
    input_specs = model.input_metadata
    inputs = {}
    for spec in input_specs:
        input_shape_dyn = spec.shape
        input_shape = [
            _batch_size if not dim else dim for dim in input_shape_dyn
        ]
        input_type = spec.dtype
        random_input = np.random.rand(*input_shape).astype(
            _get_np_dtype(input_type)
        )
        inputs[spec.name] = random_input
    return inputs


def _generate_clipvit_inputs_python():
    import numpy as np

    return {
        "pixel_values": np.random.rand(1, 3, 224, 224).astype(np.float32),
        "input_ids": np.random.rand(3, 12).astype(np.int32),
        "attention_mask": np.random.rand(3, 12).astype(np.int32),
    }


def load(model_path: Path, meta: ModelMeta):
    from max import engine

    def construct_input_specs(meta: ModelMeta) -> Sequence[engine.TensorSpec]:
        specs = []
        for name, spec in zip(meta.input_names, meta.input_specs):
            split_specs = spec.split("x")
            shape = list(map(int, split_specs[:-1]))
            dtype = (
                engine.DType.int32 if split_specs[-1]
                == "si32" else engine.DType.float32
            )
            specs.append(engine.TensorSpec(shape=shape, dtype=dtype, name=name))
        return specs

    session = engine.InferenceSession()
    print("\t Loading ...", end="")
    if meta.framework is PYTORCH:
        torch_options = engine.TorchLoadOptions()
        torch_options.input_specs = construct_input_specs(meta)
        model = session.load(model_path, torch_options)
    else:
        model = session.load(model_path)
    print("✅")
    assert model is not None, "Loading failed ❌"

    return model


@pytest.fixture(scope="module")
def test_dir():
    test_dir = tempfile.TemporaryDirectory()
    test_dir_path = Path(test_dir.name)
    yield test_dir_path

    test_dir.cleanup()


@pytest.fixture(params=top_models.keys(), scope="module")
def model_name(request):
    model_name = request.param
    model_meta = top_models[request.param]

    if (
        model_meta.framework == PYTORCH
        and TEST_ENV_OS == "main"  # main is 20.04
        and TEST_ENV_ARCH == "graviton"
    ):
        pytest.skip(
            "PyTorch via python API not supported on aarch64 + Ubuntu 20.04"
        )

    return model_name


@pytest.fixture(scope="module")
def model(model_name, test_dir):
    model_meta = top_models[model_name]

    print("Model", model_name)
    s3_uri = model_meta.s3_uri
    print("Downloading model from", s3_uri)
    model_path = download_s3_object(
        s3_uri, test_dir, model_meta.saved_model_dir_name
    )
    print("Loading and executing downloaded model ...")
    return load(model_path, model_meta)


@pytest.fixture()
def random_inputs(model, model_name):
    if "clip-vit" in model_name:
        return _generate_clipvit_inputs_python()
    else:
        return _generate_test_inputs(model)


def test_python_wheel(model, random_inputs):
    """Test the wheel which is assumed to be pip installed to the python
    environment this script is executed in"""

    output = model.execute(**random_inputs)
    assert output is not None, "Execution failed ❌"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package_path",
        help="Path to a local modular inference engine C package.",
        required=False,
    )
    args = parser.parse_args()
    if args.package_path:
        print("Testing p0 models on C package")
        c_package_test(Path(args.package_path))
    else:
        pass
