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
"""MAX config classes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import huggingface_hub

logger = logging.getLogger("max.pipelines")


class WeightPathParser:
    """Parses and validates weight paths for model configuration."""

    @staticmethod
    def parse(
        model_path: str,
        weight_path: Union[
            list[Path],
            list[str],
            tuple[Path, ...],
            tuple[str, ...],
            Path,
            str,
        ],
    ) -> tuple[list[Path], Optional[str]]:
        """Parse weight paths and extract any weights repo ID.

        Args:
            model_path: The model path to use for parsing the weight path(s)
            weight_path: The weight path(s) to parse - can be a single path, tuple, or list

        Examples:
            >>> WeightPathParser.parse("org/model", "path/to/weights.safetensors")
            (["path/to/weights.safetensors"], None)

            >>> WeightPathParser.parse("org/model", ["path/to/weights1.safetensors", "path/to/weights2.safetensors"])
            (["path/to/weights1.safetensors", "path/to/weights2.safetensors"], None)

            >>> WeightPathParser.parse("org/model", Path("path/to/weights.safetensors"))
            ([Path("path/to/weights.safetensors")], None)

            >>> WeightPathParser.parse("org/model", "org/model/weights.safetensors")
            ([Path("weights.safetensors")], "org/model")

            >>> WeightPathParser.parse("org/model", ["local_weights.safetensors", "other_org/other_model/remote_weights.safetensors"])
            ([Path("local_weights.safetensors"), Path("remote_weights.safetensors")], "other_org/other_model")

            # This is a special case where the weight path doesn't have a HF prefix,
            # which means file_exists will return False so treat the whole path
            # (without trimming a potential repo id prefix) as a local path.
            >>> WeightPathParser.parse("org/model", "very/nested/subfolder/another_nested/weights.safetensors")
            ([Path("very/nested/subfolder/another_nested/weights.safetensors")], None)

            # This on the other hand should fail, since model_path is empty
            # and we can't derive the repo id from the weight path.
            >>> WeightPathParser.parse("", "very/nested/subfolder/another_nested/weights.safetensors")
            Traceback (most recent call last):
            ...
            ValueError: Unable to derive model_path from weight_path, please provide a valid Hugging Face repository id.

        Returns:
            A tuple of (processed_weight_paths, weights_repo_id)

        Raises:
            ValueError: If weight paths are invalid or cannot be processed
        """
        # Normalize to list
        if isinstance(weight_path, tuple):
            weight_path_list = list(weight_path)
        elif not isinstance(weight_path, list):
            weight_path_list = [weight_path]
        else:
            weight_path_list = weight_path  # type: ignore

        weight_paths = []
        hf_weights_repo_id = None

        for path in weight_path_list:
            # Convert strings to Path objects and validate types
            if isinstance(path, str):
                path = Path(path)
            elif not isinstance(path, Path):
                raise ValueError(
                    "weight_path provided must either be string or Path:"
                    f" '{path}'"
                )

            # If path already exists as a file, add it directly
            if path.is_file():
                weight_paths.append(path)
                continue

            # Parse potential Hugging Face repo ID from path
            path, extracted_repo_id = (
                WeightPathParser._parse_huggingface_repo_path(model_path, path)
            )
            if extracted_repo_id:
                hf_weights_repo_id = extracted_repo_id

            weight_paths.append(path)

        return weight_paths, hf_weights_repo_id

    @staticmethod
    def _parse_huggingface_repo_path(
        model_path: str, path: Path
    ) -> tuple[Path, Optional[str]]:
        """Parse a path that may contain a Hugging Face repo ID.

        Args:
            model_path: The model path to use for parsing the weight path(s)
            path: The local path to parse HF artifacts from

        Returns:
            A tuple of (processed_path, extracted_repo_id)

        Raises:
            ValueError: If unable to derive model_path from weight_path when needed
        """
        path_pieces = str(path).split("/")

        error_message = (
            "Unable to derive model_path from weight_path, "
            "please provide a valid Hugging Face repository id."
        )
        if len(path_pieces) >= 3:
            repo_id = f"{path_pieces[0]}/{path_pieces[1]}"
            file_name = "/".join(path_pieces[2:])

            if model_path != "" and repo_id == model_path:
                return Path(file_name), None
            elif huggingface_hub.file_exists(repo_id, file_name):
                return Path(file_name), repo_id
            elif model_path == "":
                raise ValueError(error_message)
        elif model_path == "":
            raise ValueError(error_message)

        return path, None
