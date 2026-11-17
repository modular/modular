# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# TODO(GEX-4340): Delete with the legacy tile traits (GEX-4467).
#
# Each stub breaks exactly one rule of the legacy tile traits, and import
# reports them all in one run.

from max.gpu.host import DeviceContext

import extensibility
from extensibility import (
    Fused,
    Input,
    Output,
    StoreAccess,
    TileTensorable,
    _LegacyTileComputable,
    _LegacyTileComputeOutput,
    _LegacyTileStorable,
    _LegacyTileStoreAccess,
)


# A legacy tile form without `Fused`.
@extensibility.register("legacy_tile_not_fused")
struct LegacyTileNotFused:
    @staticmethod
    def execute(z: Some[Output & _LegacyTileStoreAccess], ctx: DeviceContext):
        pass


# A legacy tile form on an `Input`.
@extensibility.register("legacy_tile_on_input")
struct LegacyTileOnInput:
    @staticmethod
    def execute(
        z: Some[Output & TileTensorable],
        x: Some[Input & Fused & _LegacyTileStoreAccess],
        ctx: DeviceContext,
    ):
        pass


# The store capability on the signature.
@extensibility.register("legacy_tile_storable_on_signature")
struct LegacyTileStorableOnSignature:
    @staticmethod
    def execute(
        z: Some[Output & Fused & _LegacyTileStorable], ctx: DeviceContext
    ):
        pass


# The compute capability on the signature.
@extensibility.register("legacy_tile_computable_on_signature")
struct LegacyTileComputableOnSignature:
    @staticmethod
    def execute(
        z: Some[Output & Fused & _LegacyTileComputable & TileTensorable],
        ctx: DeviceContext,
    ):
        pass


# `TileTensorable` on the store form: the store fusion owns the write.
@extensibility.register("legacy_tile_store_tile_tensorable")
struct LegacyTileStoreTileTensorable:
    @staticmethod
    def execute(
        z: Some[Output & Fused & _LegacyTileStoreAccess & TileTensorable],
        ctx: DeviceContext,
    ):
        pass


# The compute form without `TileTensorable`.
@extensibility.register("legacy_tile_compute_no_tile_tensorable")
struct LegacyTileComputeNoTileTensorable:
    @staticmethod
    def execute(
        z: Some[Output & Fused & _LegacyTileComputeOutput], ctx: DeviceContext
    ):
        pass


# A legacy tile form together with another access form.
@extensibility.register("legacy_tile_two_access_forms")
struct LegacyTileTwoAccessForms:
    @staticmethod
    def execute(
        z: Some[Output & Fused & _LegacyTileStoreAccess & StoreAccess],
        ctx: DeviceContext,
    ):
        pass


# A valid composition on a registration that is not on the legacy tile list.
@extensibility.register("legacy_tile_not_listed")
struct LegacyTileNotListed:
    @staticmethod
    def execute(
        z: Some[Output & Fused & _LegacyTileStoreAccess], ctx: DeviceContext
    ):
        pass


# A listed name on a platform it is not listed for.
@extensibility.register("tile_rms_norm", extensibility.type("gpu"))
struct LegacyTileWrongPlatform:
    @staticmethod
    def execute(
        z: Some[Output & Fused & _LegacyTileStoreAccess], ctx: DeviceContext
    ):
        pass
