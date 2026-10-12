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

# Navigate to workspace root (git repository root)
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"


# Extract transformers version from uv.lock
VERSION=$(grep -A1 '^name = "transformers"$' oss/modular/bazel/pip/requirements/uv.lock | grep '^version' | cut -d'"' -f2)
echo "Transformers version: $VERSION"

# Clone transformers repo if not already present or if empty
if [ ! -d "local_transformers/src" ]; then
  echo "Cloning transformers v$VERSION..."
  rm -rf local_transformers
  git clone --depth 1 --branch "v$VERSION" https://github.com/huggingface/transformers.git local_transformers
  echo "✓ Cloned transformers"
else
  echo "✓ local_transformers already exists"
fi

# Create MODULE.bazel for local transformers
cat > local_transformers/MODULE.bazel << 'EOF'
module(name = "transformers_local", version = "0.0.0")

bazel_dep(name = "rules_python", version = "1.7.0")
EOF

# Create BUILD.bazel for local transformers
cat > local_transformers/BUILD.bazel << 'EOF'
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "lib",
    srcs = glob(["src/transformers/**/*.py"]),
    imports = ["src"],
    visibility = ["//visibility:public"],
)
EOF

# Add bazel_dep and local_path_override to root MODULE.bazel if not present
if ! grep -q "transformers_local" MODULE.bazel; then
  sed -i '2a\\nlocal_path_override(module_name = "transformers_local", path = "local_transformers")\nbazel_dep(name = "transformers_local", version = "0.0.0")' MODULE.bazel
fi

# Modify pip_requirement.bzl to special case transformers
cat > oss/modular/bazel/pip/pip_requirement.bzl << 'EOF'
"""Wrapper around rules_python's requirement function to handle platform selection."""

def pip_requirement(name):
    if name == "transformers":
        return "@transformers_local//:lib"
    return "@modular_pip_lock_file_repo//deps:{}".format(name)
EOF

echo "✓ Setup complete. Edit local_transformers/src/transformers as needed."
echo "✓ Run cleanup script when done."
