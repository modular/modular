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
import shutil
import subprocess
import platform
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
    "dlrm-rm2": {
        "s3_uri": "s3://modular-model-storage/dlrm/dlrm_rm2.tar.gz",
        "saved_model_dir_name": "dlrm-rm2",
        "input_names": ["ls_i", "xt"],
        "input_shapes": ["2048x40x1", "2048x256"],
        "output_names": ["output_0"],
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


def setup_environment(package_library_path: Path, package_binary_path: Path):
    for key in os.environ:
        if "MODULAR" in key:
            del os.environ[key]

    if platform.system() == "Windows":
        os.environ["Path"] = f"{os.environ['Path']};{package_binary_path}"
    else:
        os.environ["PATH"] = f"{os.environ['PATH']}:{package_binary_path}"

    os.environ["MODULAR_FRAMEWORK_ROOT"] = str(package_library_path)

    # TODO(https://github.com/modularml/modular/issues/17843)
    assert "MODULAR_DERIVED_PATH" not in os.environ


def main(package_path: Path):
    package_library_path = package_path / "lib"
    package_binary_path = package_path / "bin"
    modular_api_executor_binary = package_binary_path / "modular-api-executor"

    setup_environment(package_library_path, package_binary_path)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package_path",
        help="Path to a local modular inference engine C package.",
        required=True,
    )
    args = parser.parse_args()
    main(Path(args.package_path))
