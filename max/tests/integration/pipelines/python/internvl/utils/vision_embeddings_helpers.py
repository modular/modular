# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Helper functions for InternVL vision embeddings tests."""

import numpy as np
import torch
from internvl_impl.configuration_intern_vit import (
    InternVisionConfig as HFInternVisionConfig,
)
from internvl_impl.modeling_intern_vit import (
    InternVisionEmbeddings as HFInternVisionEmbeddings,
)
from max.driver import Accelerator, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.internvl.internvl import InternVisionEmbeddings
from max.pipelines.architectures.internvl.model_config import InternVLConfig
from max.pipelines.architectures.internvl.tokenizer import (
    extract_patches_from_image,
)
from torch.utils.dlpack import from_dlpack

from utils.weight_converter import convert_hf_to_max_weights


@torch.no_grad()
def generate_torch_outputs(
    pixel_values: torch.Tensor,
    hf_config: HFInternVisionConfig,
    embeddings_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Generate reference outputs using the vLLM implementation."""

    # Create the reference model
    ref_model = (
        HFInternVisionEmbeddings(
            hf_config,
        )
        .to(torch.bfloat16)
        .to("cuda")
    )

    # Load weights
    ref_model.load_state_dict(embeddings_weights, strict=True)
    ref_model.eval()

    # Forward pass
    with torch.no_grad():
        output = ref_model(pixel_values)

    return output


def generate_max_outputs(
    pixel_values: torch.Tensor,
    internvl_config: InternVLConfig,
    embeddings_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Generate outputs using MAX InternVisionEmbeddings implementation."""
    is_gpu = isinstance(device, Accelerator)
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()
    pixel_values = pixel_values.cuda() if is_gpu else pixel_values.cpu()

    # Convert from PyTorch's NCHW to MAX's NHWC format
    pixel_values = pixel_values.permute(0, 2, 3, 1).contiguous()

    # Extract shape information
    batch_size = pixel_values.shape[0]

    # Convert to numpy for processing
    # Convert to float32 first since numpy doesn't support bfloat16
    pixel_values_np = pixel_values.cpu().float().numpy()

    # Split each image in the batch into patches.
    all_patches = []
    for i in range(batch_size):
        img = pixel_values_np[i]
        patches = extract_patches_from_image(
            img, patch_size=internvl_config.vision_config.patch_size
        )
        all_patches.append(patches)

    # Stack all patches - shape: (batch_size, height_patches, width_patches, channels, patch_size, patch_size)
    pixel_values_np = np.stack(all_patches)

    # Convert back to torch tensor and move to device
    pixel_values = (
        torch.from_numpy(pixel_values_np).to(dtype=torch.bfloat16).cuda()
    ).contiguous()

    max_weights = convert_hf_to_max_weights(embeddings_weights)

    # Create MAX embeddings
    embeddings = InternVisionEmbeddings(internvl_config, device=device_ref)

    # Load weights
    embeddings.load_state_dict(
        state_dict=max_weights,
        strict=True,
    )

    session = InferenceSession(devices=[device])

    input_type = TensorType(dtype, shape=pixel_values.shape, device=device_ref)

    with Graph("InternVisionEmbeddings", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        output = embeddings(x.tensor)
        graph.output(output)

    compiled = session.load(graph, weights_registry=embeddings.state_dict())

    # Execute the model and get the first result
    result = compiled.execute(Tensor.from_dlpack(pixel_values).to(device))
    max_tensor = result[0]
    return from_dlpack(max_tensor)
