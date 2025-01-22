# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from hypothesis import strategies as st
from max.dtype import DType

dtypes = st.sampled_from([d for d in DType if d is not DType._unknown])
st.register_type_strategy(DType, dtypes)
