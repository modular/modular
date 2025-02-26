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


import os
import time

import pandas as pd
import streamlit as st
import torch
from max import engine
from max.dtype import DType
from shared import menu, modular_cache_dir
from transformers import BertForMaskedLM, BertTokenizer

st.set_page_config("Bert", page_icon="👓")
menu()

"""
# 👓 Bert

A basic implementation of Bert using MAX. Type a text string, using `[MASK]` to indicate where you want the model to predict a word.
"""

HF_MODEL_NAME = "bert-base-uncased"


# If batch, seq_len, or mlm options change, recompile the torchscript.
@st.cache_data
def compile_torchscript(batch: int, seq_len: int):
    model = BertForMaskedLM.from_pretrained(HF_MODEL_NAME)
    input_dict = {
        "input_ids": torch.zeros((batch, seq_len), dtype=torch.int64),
        "attention_mask": torch.zeros((batch, seq_len), dtype=torch.int64),
        "token_type_ids": torch.zeros((batch, seq_len), dtype=torch.int64),
    }
    model.eval()
    model.config.return_dict = False
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model, example_kwarg_inputs=dict(input_dict), strict=False
        )
    torch.jit.save(traced_model, model_path)


@st.cache_resource(show_spinner="Starting MAX Bert Inference Session")
def max_bert_session(model_path: str, batch: int, seq_len: int):
    # Wait short time for spinner to start correctly
    time.sleep(1)
    session = engine.InferenceSession()
    inputs = [
        torch.zeros((batch, seq_len), dtype=torch.int64),
        torch.zeros((batch, seq_len), dtype=torch.int64),
        torch.zeros((batch, seq_len), dtype=torch.int64),
    ]
    input_spec_list = [
        engine.TorchInputSpec(shape=tensor.size(), dtype=DType.int64)
        for tensor in inputs
    ]
    return session.load(model_path, input_specs=input_spec_list)


@st.cache_data()
def get_tokenizer():
    return BertTokenizer.from_pretrained(HF_MODEL_NAME)


def softmax(logits):
    exp_logits = torch.exp(logits - torch.max(logits))
    return exp_logits / exp_logits.sum(dim=-1, keepdim=True)


model_state = st.empty()

show_predictions = st.sidebar.checkbox("Show top 5 predictions", True)
model_path = st.sidebar.text_input(
    "Model Path",
    os.path.join(modular_cache_dir(), "bert-mlm.torchscript"),
)
batch = st.sidebar.number_input("Batch Size", 1, 64)
seq_len = st.sidebar.slider("Sequence Length", 128, 1024)
input_text = st.text_input("Text Input", "Don't [MASK] about it")

compile_torchscript(batch, seq_len)

if st.button("Predict Word"):
    masks = input_text.split("[MASK]")
    if len(masks) > 2:
        st.error("Cannot have more than a single [MASK] in the input text")
        exit(1)
    if len(masks) < 2:
        st.error("Require at least one [MASK] in the input text")
        exit(1)

    model = max_bert_session(model_path, batch, seq_len)

    tokenizer = get_tokenizer()
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )

    masked_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(
        as_tuple=True
    )[1]

    outputs = model.execute_legacy(**inputs)["result0"]

    logits = torch.from_numpy(outputs[0, masked_index, :])

    predicted_token_id = logits.argmax(dim=-1)
    predicted_tokens = tokenizer.decode(
        [predicted_token_id],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    st.text_area(
        "Filled Mask", input_text.replace("[MASK]", predicted_tokens, 40)
    )

    if show_predictions:
        # Get top N predictions for the [MASK] token
        top_n = 5
        mask_logits = logits.squeeze(0)  # Remove batch dimension
        top_n_probs, top_n_indices = torch.topk(softmax(mask_logits), top_n)

        top_n_tokens = tokenizer.convert_ids_to_tokens(top_n_indices.tolist())
        top_n_probs_percent = [prob * 100 for prob in top_n_probs.tolist()]

        # Create a dictionary for the bar chart data
        data = {"Token": top_n_tokens, "Probability (%)": top_n_probs_percent}

        # Create a DataFrame for the table
        top_n_df = pd.DataFrame(data)

        st.write("Top N predictions for [MASK]:")
        st.bar_chart(top_n_df.set_index("Token"))  # Display bar chart!
