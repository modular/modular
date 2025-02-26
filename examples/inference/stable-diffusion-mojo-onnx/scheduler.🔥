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
#
# This code is directly ported from the hugging face diffusers library.
#
# Copyright 2024 Zhejiang University Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""A simple helper implementation of a pndm schedular for stable diffusion """

from max.tensor import Tensor

alias FloatTensor = Tensor[DType.float32]


struct Scheduler:
    alias training_steps = 1000
    alias init_noise_sigma = 1.0
    alias beta_start = 0.00085
    alias beta_end = 0.012

    var counter: Int
    var step_ratio: Int
    var timesteps: List[Int]
    var alphas_cumprod: List[Float32]
    var previous_outputs: List[FloatTensor]

    def __init__(out self, inference_steps: Int):
        self.counter = 0
        self.previous_outputs = List[FloatTensor]()
        self.alphas_cumprod = List[Float32]()

        # Generate timesteps.
        self.step_ratio = Self.training_steps // inference_steps
        self.timesteps = List[Int]()
        for i in range(1, Self.training_steps, self.step_ratio):
            self.timesteps.append(i)

        self.timesteps.reverse()

        # Duplicate the second timestep because this scheduler skips the prk steps.
        if len(self.timesteps) >= 2:
            self.timesteps.insert(1, self.timesteps[1])

        # Generate alpha cumulative product.
        var start = Self.beta_start**0.5
        var end = Self.beta_end**0.5
        var step = (end - start) / Self.training_steps
        var cumprod = 1.0
        for i in range(Self.training_steps):
            var beta = start + (end - start) * (i / Self.training_steps)
            var alpha = 1.0 - (beta**2)
            cumprod *= alpha
            self.alphas_cumprod.append(cumprod.cast[DType.float32]())

    def step(
        mut self,
        model_output: FloatTensor,
        timestep: Int,
        sample: FloatTensor,
    ) -> FloatTensor:
        var previous_timestep = timestep - self.step_ratio

        if self.counter == 1:
            # Special case: the second timestep is repeated.
            previous_timestep = timestep
            timestep = timestep + self.step_ratio
        else:
            # TODO: clear out old tensors.
            # For some reason, doing this leads to a crash.
            # self.previous_outputs = self.previous_outputs[-3:]
            self.previous_outputs.append(FloatTensor(model_output))

        averaged_output = FloatTensor(model_output)
        if len(self.previous_outputs) == 1 and self.counter == 0:
            # First input, there is no data to average.
            pass
        elif len(self.previous_outputs) == 1 and self.counter == 1:
            averaged_output = (
                averaged_output + self.previous_outputs[-1]
            ) / 2.0
        elif len(self.previous_outputs) == 2:
            averaged_output = (
                3 * self.previous_outputs[-1] - self.previous_outputs[-2]
            ) / 2.0
        elif len(self.previous_outputs) == 3:
            averaged_output = (
                23 * self.previous_outputs[-1]
                - 16 * self.previous_outputs[-2]
                + 5 * self.previous_outputs[-3]
            ) / 12.0
        else:
            averaged_output = (1.0 / 24.0) * (
                55 * self.previous_outputs[-1]
                - 59 * self.previous_outputs[-2]
                + 37 * self.previous_outputs[-3]
                - 9 * self.previous_outputs[-4]
            )

        previous_sample = self._get_previous_sample(
            sample, timestep, previous_timestep, averaged_output
        )
        self.counter += 1

        return previous_sample

    def _get_previous_sample(
        mut self,
        sample: FloatTensor,
        timestep: Int,
        previous_timestep: Int,
        model_output: FloatTensor,
    ) -> FloatTensor:
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_previous = (
            self.alphas_cumprod[previous_timestep] if previous_timestep
            >= 0 else self.alphas_cumprod[0]
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_previous = 1 - alpha_prod_t_previous

        sample_coeff = (alpha_prod_t_previous / alpha_prod_t) ** (0.5)

        model_output_denom_coeff = alpha_prod_t * beta_prod_t_previous ** (
            0.5
        ) + (alpha_prod_t * beta_prod_t * alpha_prod_t_previous) ** (0.5)

        previous_sample = (
            sample_coeff * sample
            - (alpha_prod_t_previous - alpha_prod_t)
            * model_output
            / model_output_denom_coeff
        )
        return previous_sample
