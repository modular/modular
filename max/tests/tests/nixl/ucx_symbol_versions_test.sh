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
#
# The verbs UCX plugins link with `-Wl,-z,undefs`, so nothing about their
# rdma-core symbols can fail the link. Unversioned ones bind to the oldest
# compat definition. For ibv_* that is IBVERBS_1.0, whose struct ibv_device ABI
# differs: UCX enumerates zero RDMA devices. For mlx5dv_init_obj it is
# MLX5_1.0, which hands UCX a pointer into the device context instead of the CQ
# doorbell page: UCX's doorbell writes corrupt the context, and closing it
# aborts the process. Only a repin of UCX or rdma-core can reintroduce this,
# which is exactly when nobody is looking for it.

set -euo pipefail

readonly readelf="$1"
shift

status=0
for plugin in "$@"; do
  unversioned=$("$readelf" --dyn-symbols "$plugin" |
    awk 'NF > 1 && $(NF-1) == "UND" && $NF ~ /^(ibv_|mlx5dv_)/ && $NF !~ /@/ { print $NF }')

  if [[ -n "$unversioned" ]]; then
    echo "error: $plugin has undefined rdma-core symbols with no version tag," \
         "so they will bind to the wrong ABI:" >&2
    echo "$unversioned" >&2
    status=1
  fi
done

exit "$status"
