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
"""Universal MAX provider options applicable to all requests."""

from pydantic import BaseModel, ConfigDict, Field


class MaxProviderOptions(BaseModel):
    """Universal MAX provider options.

    These options are applicable to all requests regardless of pipeline or modality.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_endpoint: str | None = Field(
        None,
        description=(
            "Optional target endpoint identifier for routing the request to a specific "
            "service or model instance. This should be used in disaggregate serving "
            "scenarios, when you want to dynamically route to a specific instance. "
            "If not specified, the request will be routed to the default endpoint."
        ),
    )
