# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from pathlib import Path
from sys import version_info
from enum import Enum
from typing import Union, Optional

if version_info.minor <= 8:
    from typing import List
else:
    List = list

class Model:
    def execute(self, *args) -> None: ...
    def init(self) -> None: ...

class InferenceSession:
    def __init__(self, config: dict = ...) -> None: ...
    def load(
        self, model_path: Union[str, Path], config: dict = ...
    ) -> Model: ...

class TensorSpec:
    def shape(self) -> List[int]: ...

class DType(Enum): ...
