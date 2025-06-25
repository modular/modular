# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""JSON encoder and decoder that are compatible with numpy."""

import base64
from json import JSONDecoder, JSONEncoder

import numpy as np


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__np__": base64.b64encode(obj.tobytes()).decode("ascii"),
                "shape": obj.shape,
                "dtype": str(obj.dtype),
            }
        elif isinstance(obj, np.generic):
            return obj.item()
        return JSONEncoder.default(self, obj)


class NumpyDecoder(JSONDecoder):
    def __init__(self, *args, **kwargs) -> None:
        JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs
        )

    def object_hook(self, dct):
        if "__np__" in dct:
            shape = dct["shape"]
            dtype = np.dtype(dct["dtype"])
            return np.frombuffer(
                base64.b64decode(dct["__np__"]), dtype=dtype
            ).reshape(shape)
        return dct
