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

import numpy as np
import torch


def post_process():
    logits = torch.from_numpy(np.fromfile("outputs.bin", dtype=np.float32))
    predictions = torch.argmax(logits, dim=-1)

    predicted_label = predictions.item()
    sentiment_labels = {0: "Negative", 1: "Positive"}
    print(f"Predicted sentiment: {sentiment_labels[predicted_label]}")


if __name__ == "__main__":
    torch.set_default_device("cpu")
    post_process()
