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

# Baseline accuracy windows for perfsect regression detection.
#
# Floor formula: MAX(window_start → commit_date) - tolerance
#
# The window grows as commits accumulate; MAX rises with model improvements.
# tolerances are fixed per (model, eval_task), derived once from
# window_start → 2026-05-22 as MAX - MIN over that period, ensuring all
# samples in that calibration window pass.
#
# Models without tolerances are skipped for status computation entirely.
#
# status field:
#   (absent)            - window is active, tolerances required
#   "insufficient_data" - too few runs to be reliable; status skipped
#   "no_data"           - no historical runs exist yet; status skipped

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AccuracyWindow:
    window_start: str | None = None
    status: Literal["insufficient_data", "no_data"] | None = None
    tolerances: dict[str, float] = field(default_factory=dict)


# fmt: off
ACCURACY_WINDOWS: dict[str, AccuracyWindow] = {
    # Stable models — full window from 2026-01-01
    "allenai/Olmo-3-7B-Instruct":                               AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.05625}),
    "deepseek-ai/DeepSeek-R1-0528":                             AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.0375}),
    "deepseek-ai/DeepSeek-V2-Lite-Chat":                        AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.103125}),
    "deepseek-ai/DeepSeek-V3.1-Terminus":                       AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.021875}),
    "google/gemma-3-1b-it":                                     AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.05}),
    "google/gemma-4-31B-it":                                    AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.078125, "chartqa": 0.009375}),
    "lukealonso/MiniMax-M2.7-NVFP4":                            AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.028125}),
    "meta-llama/Llama-3.1-8B-Instruct":                        AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.075}),
    "microsoft/Phi-3.5-mini-instruct":                          AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.0375}),
    "microsoft/phi-4":                                          AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.021875}),
    "MiniMaxAI/MiniMax-M2.7":                                   AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.0125}),
    "nvidia/DeepSeek-V3.1-NVFP4":                               AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.10625}),
    "nvidia/Gemma-4-31B-IT-NVFP4":                              AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.003125, "chartqa": 0.015625}),
    "nvidia/Llama-3.1-405B-Instruct-NVFP4":                    AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.025}),
    "openai/gpt-oss-20b":                                       AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.034375}),
    "Qwen/Qwen2.5-7B-Instruct":                                 AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.046875}),
    "Qwen/Qwen3-235B-A22B-Instruct-2507":                       AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.015625}),
    "Qwen/Qwen3-30B-A3B-Instruct-2507":                        AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.01875}),
    "Qwen/Qwen3.5-9B":                                          AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.0125, "chartqa": 0.01875}),
    "stepfun-ai/Step-3.5-Flash":                                AccuracyWindow("2026-01-01", tolerances={"gsm8k_cot_llama": 0.021875}),

    # Post-dip windows — start after last known instability
    "allenai/olmOCR-2-7B-1025-FP8":                            AccuracyWindow("2026-04-01", tolerances={"gsm8k_cot_llama": 0.05, "chartqa": 0.065625}),
    "google/gemma-3-27b-it":                                    AccuracyWindow("2026-05-01", tolerances={"gsm8k_cot_llama": 0.01875, "chartqa": 0.01875}),
    "mistralai/Mistral-Nemo-Instruct-2407":                     AccuracyWindow("2026-04-01", tolerances={"gsm8k_cot_llama": 0.025}),
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503":            AccuracyWindow("2026-05-01", tolerances={"gsm8k_cot_llama": 0.021875}),
    "modularai/Llama-3.1-405B-Instruct-autofp8":               AccuracyWindow("2026-05-01", tolerances={"gsm8k_cot_llama": 0.01875}),
    "nvidia/Kimi-K2.5-NVFP4":                                   AccuracyWindow("2026-05-01", tolerances={"gsm8k_cot_llama": 0.009375, "chartqa": 0.0625}),
    "OpenGVLab/InternVL3_5-8B-Instruct":                       AccuracyWindow("2026-02-01", tolerances={"gsm8k_cot_llama": 0.034375, "chartqa": 0.034375}),
    "Qwen/Qwen2.5-VL-7B-Instruct":                             AccuracyWindow("2026-02-01", tolerances={"gsm8k_cot_llama": 0.05625, "chartqa": 0.04375}),
    "Qwen/Qwen3-8B":                                            AccuracyWindow("2026-04-01", tolerances={"gsm8k_cot_llama": 0.034375}),
    "Qwen/Qwen3-VL-4B-Instruct":                               AccuracyWindow("2026-05-01", tolerances={"gsm8k_cot_llama": 0.025, "chartqa": 0.034375}),
    "Qwen/Qwen3-VL-4B-Instruct-FP8":                           AccuracyWindow("2026-05-01", tolerances={"gsm8k_cot_llama": 0.0125, "chartqa": 0.025}),
    "Qwen/Qwen3-VL-30B-A3B-Thinking":                          AccuracyWindow("2026-02-01", tolerances={"gsm8k_cot_llama": 0.05, "chartqa": 0.084375}),
    "RedHatAI/gemma-3-27b-it-FP8-dynamic":                     AccuracyWindow("2026-05-01", tolerances={"gsm8k_cot_llama": 0.025, "chartqa": 0.028125}),
    "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic":       AccuracyWindow("2026-05-01", tolerances={"gsm8k_cot_llama": 0.084375}),
    "unsloth/gpt-oss-20b-BF16":                                 AccuracyWindow("2026-03-01", tolerances={"gsm8k_cot_llama": 0.034375}),

    # Insufficient data — window defined but too few runs to be reliable
    "amd/Kimi-K2.5-MXFP4":                                     AccuracyWindow("2026-05-01", status="insufficient_data"),
    "amd/MiniMax-M2.7-MXFP4":                                   AccuracyWindow("2026-05-01", status="insufficient_data"),
    "ByteDance-Seed/academic-ds-9B":                            AccuracyWindow("2026-01-01", status="insufficient_data"),
    "google/gemma-4-26B-A4B-it":                                AccuracyWindow("2026-04-01", status="insufficient_data"),
    "Qwen/Qwen3.6-27B":                                         AccuracyWindow("2026-05-01", status="insufficient_data"),

    # No data — no historical runs exist yet
    "nvidia/Gemma-4-26B-A4B-NVFP4":                            AccuracyWindow(status="no_data"),
}
# fmt: on
