# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from pathlib import Path
from sys import version_info

if version_info.minor <= 8:
    from typing import List
else:
    List = list

class Model:
    def execute(self, *args) -> None: ...
    def load(self) -> None: ...

class InferenceSession:
    def __init__(self, config: dict = ...) -> None: ...
    def compile(self, model_path: Path, config: dict = ...) -> Model: ...

class TensorSpec:
    def shape(self) -> List[int]: ...
