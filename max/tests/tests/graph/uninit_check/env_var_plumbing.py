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
"""Verifies InferenceSession plumbing for the uninitialized-read-check config.

Used as a subprocess by test_uninit_check_e2e.py.  When
MODULAR_MAX_DEBUG_UNINITIALIZED_READ_CHECK=true is in the environment (or
the `max-debug.uninitialized-read-check` Config key is set),
InferenceSession.__init__ should set MODULAR_DEBUG_DEVICE_ALLOCATOR to
include "uninitialized-poison".
"""

import os

from max.driver import CPU
from max.engine import InferenceSession

session = InferenceSession(devices=[CPU()])

allocator = os.environ.get("MODULAR_DEBUG_DEVICE_ALLOCATOR", "")
if "uninitialized-poison" in allocator:
    print("ALLOCATOR_SET")
else:
    print(f"ALLOCATOR_NOT_SET (value={allocator!r})")

print("PLUMBING_OK")
