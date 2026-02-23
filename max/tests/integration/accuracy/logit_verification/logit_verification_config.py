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

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Agent(BaseModel):
    "Kiteworks agent configuration"

    pool: str
    arch: str | None = None
    resource_class: str
    queue: str | None = None


class PipelineConfig(BaseModel):
    "Logit verification pipeline configuration"

    pre_submit_agents: list[Agent]
    pipeline: str


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
