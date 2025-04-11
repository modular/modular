# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .module import PackageTrait, PackageTrait2


trait PackageChildTrait(PackageTrait, PackageTrait2):
    pass
