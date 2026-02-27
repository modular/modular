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

"""Pydantic models for logit verification pipeline configuration.

Reads and validates config.yaml, which is the single source of truth for all
logit verification pipeline definitions. The config contains agent runner
configurations used to matrix logit verification over CI runs.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

SupportedEncoding = Literal[
    "float32",
    "bfloat16",
    "q4_k",
    "q4_0",
    "q6_k",
    "float8_e4m3fn",
    "float4_e2m1fnx2",
    "gptq",
]


class DeviceKind(enum.Enum):
    CPU = "cpu"
    GPU = "gpu"


class PregeneratedTorchGoldens(BaseModel):
    """Paths to pregenerated torch golden logits."""

    tar_file: str
    """S3 path to the tar file containing the bundled golden json files."""
    json_file: str
    """Name of the json file containing the golden logits."""


class Agent(BaseModel):
    "Kiteworks agent configuration"

    pool: str
    arch: str | None = None
    resource_class: str
    queue: str | None = None


class PipelineConfig(BaseModel):
    "Logit verification pipeline configuration"

    pre_submit_agents: list[Agent] = Field(default_factory=list)
    pipeline: str

    compatible_with: list[DeviceKind] = Field(default_factory=list)
    encoding: SupportedEncoding
    tags: list[str] = Field(default_factory=list)
    pregenerated_torch_goldens: PregeneratedTorchGoldens | None = None

    absolute_tolerance: float | None = None
    relative_tolerance: float | None = None
    cos_dist_threshold: float | None = None
    kl_div_threshold: float | None = None
    timeout: int | None = None

    ssim_threshold: float | None = None
    lpips_threshold: float | None = None


class LogitVerificationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pipelines: dict[str, PipelineConfig] = Field(
        alias="logit_verification_pipelines"
    )

    @property
    def pre_submit_matrix(self) -> list[list[tuple[str, Agent]]]:
        return [
            [
                (pipeline_name, agent)
                for agent in self.pipelines[pipeline_name].pre_submit_agents
            ]
            for pipeline_name in self.pipelines
        ]

    @classmethod
    def from_file(
        cls,
        path: Path = Path(__file__).parent / "logit_verification_config.yaml",
    ) -> LogitVerificationConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return LogitVerificationConfig(**raw)


LOGIT_VERIFICATION_CONFIG = LogitVerificationConfig.from_file()
