# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Mojo APIs to run inference with MAX Engine.

Although there are several modules in this `max.engine` package, you'll get
everything you need from this top-level `engine` namespace, so you don't need
to import each module.

For example, the basic code you need to run an inference looks like this:

```mojo
from max import engine

fn main() raises:
    # Load your model:
    var session = engine.InferenceSession()
    var model = session.load(model_path)

    # Get the inputs, then run an inference:
    var outputs = model.execute(inputs)
    # Process the outputs here.
```

For details, see how to [run inferencew with Mojo](/engine/mojo/get-started).
"""

from .info import get_version
from .model import Model
from .session import InferenceSession, SessionOptions, TorchLoadOptions
from .shape_element import ShapeElement
from .tensor_spec import EngineTensorSpec
from .tensor_map import TensorMap
from .tensor import EngineTensorView, EngineNumpyView, NamedTensor
from .value import Value, List
