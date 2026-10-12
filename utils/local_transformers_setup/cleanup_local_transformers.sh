#!/bin/bash
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

set -euo pipefail

# Navigate to workspace root
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# Restore pip_requirement.bzl
cat > oss/modular/bazel/pip/pip_requirement.bzl << 'EOF'
"""Wrapper around rules_python's requirement function to handle platform selection."""

def pip_requirement(name):
    return "@modular_pip_lock_file_repo//deps:{}".format(name)
EOF
echo "✓ Restored pip_requirement.bzl"

# Remove transformers_local from root MODULE.bazel
# Create a temp file to build the cleaned version
python3 << 'PYTHON_SCRIPT'
import re

with open('MODULE.bazel', 'r') as f:
    content = f.read()

# Remove multi-line local_path_override block containing transformers_local
pattern = r'local_path_override\([^)]*transformers_local[^)]*\)\n*'
content = re.sub(pattern, '', content, flags=re.DOTALL)

# Remove single-line bazel_dep for transformers_local
pattern = r'bazel_dep\(name = "transformers_local"[^\n]*\)\n*'
content = re.sub(pattern, '', content)

# Clean up multiple consecutive empty lines (keep at most one)
content = re.sub(r'\n\n\n+', '\n\n', content)

with open('MODULE.bazel', 'w') as f:
    f.write(content)
PYTHON_SCRIPT

echo "✓ Removed transformers_local from MODULE.bazel"

# Remove local_transformers directory if it exists
if [ -d local_transformers ]; then
  rm -rf local_transformers
  echo "✓ Removed local_transformers directory"
fi

echo "✓ Cleanup complete"
