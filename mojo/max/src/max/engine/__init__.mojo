# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""APIs to run inference with MAX Engine.

Although there are several modules in this `max.engine` package, you'll get
everything you need from this top-level `engine` namespace, so you don't need
to import each module.

For example, the basic code you need to run an inference looks like this:

```mojo
from max import engine

def main():
    # Load your model:
    var session = engine.InferenceSession()
    var model = session.load(model_path)

    # Get the inputs, then run an inference:
    var outputs = model.execute(inputs)
    # Process the outputs here.
```
"""

from ._context import PrintStyle
from .info import get_version
from .model import Model
from .session import InferenceSession, InputSpec, SessionOptions
from .shape_element import ShapeElement
from .tensor import EngineNumpyView, EngineTensorView, NamedTensor
from .tensor_map import TensorMap
from .tensor_spec import EngineTensorSpec
from .value import List, Value
