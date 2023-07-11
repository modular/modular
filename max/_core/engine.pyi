# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from pathlib import Path
from sys import version_info
from enum import Enum
from typing import Union, Optional

import numpy as np

if version_info.minor <= 8:
    from typing import Dict, List
else:
    Dict = dict
    List = list

class Model:
    def execute(self, **kwargs) -> Dict[str, np.ndarray]: ...
    def init(self) -> None: ...

class InferenceSession:
    def __init__(self, num_threads: Optional[int] = ...) -> None: ...
    def load(
        self, model_path: Union[str, Path], config: dict = ...
    ) -> Model: ...

class TensorSpec:
    def shape(self) -> List[int]: ...

class DType(Enum): ...
