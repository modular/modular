##===----------------------------------------------------------------------===##
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
##===----------------------------------------------------------------------===##

# shellcheck disable=SC2034  # `exclude` is consumed when this file is sourced.

# Failures vLLM 0.31.0 serving the same checkpoint with its glm45 parsers
# shares, from the full validation run of 2026-10-26 and the vLLM run of
# 2026-10-27.
_glm53_flash_vllm_failures=(
  # tool_schema_validation: the model writes the value the prompt asks for,
  # and the glm_xml structural tag MAX enforces it with checks types, enums and
  # patterns but not these keywords (vLLM: FAIL each, 11 of the scenario's
  # tests, including the 5 MAX enforces).
  'tool_schema_validation:enforce_missing_required'
  'tool_schema_validation:enforce_additional_property'
  'tool_schema_validation:enforce_number_over_maximum'
  'tool_schema_validation:enforce_max_length'
  'tool_schema_validation:enforce_format_email'
  'tool_schema_validation:enforce_min_properties'
  # connection_exhaustion: HTTP/1.1 pipelining is not honored by either
  # server's ASGI stack (vLLM: FAIL, error=None).
  'connection_exhaustion:pipelined_requests'
)

# Join into the comma-separated form the --exclude flag expects.
exclude=$(
  IFS=,
  printf "%s" "${_glm53_flash_vllm_failures[*]}"
)
