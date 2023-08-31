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
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

MODULAR_PATH = Path.cwd()
MODELS_DIR = MODULAR_PATH / "test-models"
YAML_PATH = MODULAR_PATH / "Models"


top_models = {
    "bert-base-uncased-seqlen-128": {
        "s3_uri": "s3://modular-model-storage/Tensorflow/bert-base-uncased_dynamic_seqlen_savedmodel.tar.gz",
        "saved_model_dir_name": "bert-base-uncased_dynamic_seqlen",
        "input_names": ["attention_mask", "input_ids", "token_type_ids"],
        "input_shapes": ["1x128", "1x128", "1x128"],
        "output_names": ["last_hidden_state", "pooler_output"],
    },
    "dlrm-rm1-multihot": {
        "s3_uri": "s3://modular-model-storage/dlrm/dlrm-facebook-1.tar.gz",
        "saved_model_dir_name": "dlrm-facebook-1",
        "input_names": ["ls_i", "xt"],
        "input_shapes": ["4096x8x100", "4096x256"],
        "output_names": ["output_0"],
    },
    "dlrm-rm1": {
        "s3_uri": "s3://modular-model-storage/dlrm/dlrm_rm1.tar.gz",
        "saved_model_dir_name": "dlrm-rm1",
        "input_names": ["ls_i", "xt"],
        "input_shapes": ["16384x8x1", "16384x256"],
        "output_names": ["output_0"],
    },
    "dlrm-rm2": {
        "s3_uri": "s3://modular-model-storage/dlrm/dlrm_rm2.tar.gz",
        "saved_model_dir_name": "dlrm-rm2",
        "input_names": ["ls_i", "xt"],
        "input_shapes": ["2048x40x1", "2048x256"],
        "output_names": ["output_0"],
    },
    "dlrm-rm3": {
        "s3_uri": "s3://modular-model-storage/dlrm/dlrm-facebook-3.tar.gz",
        "saved_model_dir_name": "dlrm-rm3",
        "input_names": ["ls_i", "xt"],
        "input_shapes": ["1024x10x1", "1024x2560"],
        "output_names": ["output_0"],
    },
    "clip-vit-large-patch14": {
        "s3_uri": "s3://modular-model-storage/Tensorflow/clip_vit_large_patch14_savedmodel.tar.gz",
        "saved_model_dir_name": "clip_vit_large_patch14_savedmodel",
        "input_names": ["pixel_values", "input_ids", "attention_mask"],
        "input_shapes": ["1x3x224x224", "1x16", "1x16"],
        "output_names": [
            "image_embeds",
            "logits_per_image",
            "logits_per_text",
            "text_embeds",
            "text_model_output_last_hidden_state",
            "text_model_output_pooler_output",
            "vision_model_output_last_hidden_state",
            "vision_model_output_pooler_output",
        ],
    },
}


def download_s3_object(s3_src: str, local_dst: Path):
    """Downloads and extract .tar.gz object at `s3_src` to file at `local_dst`"""

    subprocess.run(
        ["aws", "s3", "cp", f"{s3_src}", f"{local_dst}"],
        check=True,
        capture_output=True,
    )
    filename = s3_src.split("/")[-1]

    with tarfile.open(local_dst / filename, "r") as tar:
        tar.extractall(local_dst)


def remove_modular_env_variables():
    for key in os.environ:
        if "MODULAR" in key:
            del os.environ[key]

    # TODO(https://github.com/modularml/modular/issues/17843)
    assert "MODULAR_DERIVED_PATH" not in os.environ


def setup_environment_for_c(
    package_library_path: Path, package_binary_path: Path
):
    remove_modular_env_variables()

    if platform.system() == "Windows":
        os.environ["Path"] = f"{os.environ['Path']};{package_binary_path}"
    else:
        os.environ["PATH"] = f"{os.environ['PATH']}:{package_binary_path}"

    os.environ["MODULAR_FRAMEWORK_ROOT"] = str(package_library_path)


def test_c_package(package_path: Path):
    package_library_path = package_path / "lib"
    package_binary_path = package_path / "bin"
    modular_api_executor_binary = package_binary_path / "modular-api-executor"

    setup_environment_for_c(package_library_path, package_binary_path)

    with tempfile.TemporaryDirectory() as test_dir:
        test_dir_path = Path(test_dir)

        common_modular_api_executor_args = (
            "--model-inputs=zeros --allocator=malloc --num-threads=0"
            " --num-runs=1 --result-output-style=compact".split()
        )
        for key, value in top_models.items():
            download_s3_object(value["s3_uri"], test_dir_path)
            input_names_args = [
                f"--input-names={name}" for name in value["input_names"]
            ]
            output_names_args = [
                f"--output-names={name}" for name in value["output_names"]
            ]
            input_shapes_args = [
                f"--input-shapes={shape}" for shape in value["input_shapes"]
            ]
            saved_model_dir = test_dir_path / value["saved_model_dir_name"]
            cmd = [
                str(modular_api_executor_binary),
                *input_names_args,
                *input_shapes_args,
                *output_names_args,
                *common_modular_api_executor_args,
                saved_model_dir,
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

    from modular import engine

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


def generate_test_inputs(model):
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


def generate_clipvit_inputs_python():
    import numpy as np

    return {
        "pixel_values": np.random.rand(1, 3, 224, 224).astype(np.float32),
        "input_ids": np.random.rand(3, 12).astype(np.int32),
        "attention_mask": np.random.rand(3, 12).astype(np.int32),
    }


def load_and_execute_on_random_input(model_path: Path):
    from modular import engine

    session = engine.InferenceSession()
    print("\t Loading ...", end="")
    model = session.load(model_path)
    print("✅")
    assert model is not None, "Loading failed ❌"
    if "clip_vit" in str(model_path):
        inputs = generate_clipvit_inputs_python()
    else:
        inputs = generate_test_inputs(model)
    print("\t Executing ...", end="")
    output = model.execute(**inputs)
    print("✅")
    assert output is not None, "Execution failed ❌"


def test_python_wheel():
    """Test the wheel which is assumed to be pip installed to the python
    environment this script is executed in"""
    remove_modular_env_variables()
    from modular import engine

    test_dir = tempfile.TemporaryDirectory()
    test_dir_path = Path(test_dir.name)

    session = engine.InferenceSession()
    for key, value in top_models.items():
        print("Model", key)
        s3_uri = value["s3_uri"]
        print("Downloading model from", s3_uri)
        download_s3_object(s3_uri, test_dir_path)
        model_path = test_dir_path / value["saved_model_dir_name"]
        print("Loading and executing downloaded model ...")
        load_and_execute_on_random_input(model_path)
        print("Model executed successfully ✅")
    test_dir.cleanup()


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
        test_c_package(Path(args.package_path))
    else:
        print("Testing p0 models on Python wheel")
        test_python_wheel()
