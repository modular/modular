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

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def set_seeds(seed=42):
    """Sets seed on CPU"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seeds()


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    input_size = 28 * 28
    hidden_size = 512
    num_classes = 10
    num_epochs = 5
    learning_rate = 0.001

    preprocess = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=preprocess, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=preprocess, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=128, shuffle=False
    )

    model = Model(input_size, hidden_size, num_classes)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28 * 28)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step"
                    f" [{i + 1}/{total_steps}], Loss: {loss.item():.4f}"
                )

    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            predicted = torch.argmax(probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            "Accuracy of the network on the 10000 test images:"
            f" {100 * correct / total} %"
        )

    # save weights in numpy binary format
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()

    np.save("model_weights.npy", weights)
    return


if __name__ == "__main__":
    main()
