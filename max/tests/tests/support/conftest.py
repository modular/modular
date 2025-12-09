# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from hypothesis import Phase, settings

settings.register_profile(
    "failfast", phases=[Phase.explicit, Phase.reuse, Phase.generate]
)
