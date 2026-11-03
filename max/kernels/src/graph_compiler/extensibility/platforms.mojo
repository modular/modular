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
"""Names for the platform predicates that kernel registrations use most.

Each alias is a `PlatformPredicate` built from the atoms in
`extensibility.platform_predicates`. Pass one to `@extensibility.register` as
`cond`, or combine aliases with `&` and `|`. Name a more specific platform
predicate with `comptime` where it is used:

```mojo
comptime B200 = ACCELERATOR & CUDA & model("NVIDIA B200")
```
"""

from .platform_predicates import api, type

comptime CPU = type("cpu")

# The runtime gives the label "gpu" to every accelerator, including those that
# are not GPUs.
comptime ACCELERATOR = type("gpu")

comptime CUDA = ACCELERATOR & api("cuda")

comptime HIP = ACCELERATOR & api("hip")

comptime METAL = ACCELERATOR & api("metal")
