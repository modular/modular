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
import asyncio
import os
import time
from pathlib import Path

import streamlit as st
import torch
from max.driver import CPU, Accelerator
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    SupportedEncoding,
)
from max.pipelines.architectures.llama3.config import get_llama_huggingface_file
from shared import (
    RAG_PROMPT,
    RAG_SYSTEM_PROMPT,
    hf_streamlit_download,
    load_embed_docs,
    menu,
    stream_output,
)

st.set_page_config(page_title="Llama3.1", page_icon="🦙")

"""
# Llama3.1 🦙

Compile and cache Llama3.1 built with MAX graphs so you can continuously
chat with it.

Tick `Activate RAG` on the sidebar to augment your prompts with
text from documents in the `examples/gui/ragdata` folder in format: `.txt`
`.pdf` `.csv` `.docx` `.epub` `.ipynb` `.md` `.html`.
"""

menu()


@st.cache_resource(show_spinner=False)
def start_llama3(
    weight_path: str,
    quantization: SupportedEncoding,
    max_length: int,
    max_new_tokens: int,
    use_gpu: bool,
) -> Llama3:
    config = PipelineConfig(
        architecture="LlamaForCausalLM",
        device=Accelerator() if use_gpu else CPU(),
        weight_path=[Path(weight_path)],
        quantization_encoding=quantization,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
    )
    _, pipeline = PIPELINE_REGISTRY.retrieve_factory(config)
    return pipeline


def messages_to_llama3_prompt(messages: list[dict[str, str]]) -> str:
    prompt_string = "<|begin_of_text|>"
    for message in messages:
        prompt_string += (
            f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
        )
        prompt_string += f"{message['content']}<|eot_id|>\n"
    prompt_string += "<|start_header_id|>assistant<|end_header_id|>"
    return prompt_string


if torch.cuda.is_available():
    use_gpu = st.sidebar.checkbox(
        f"Use GPU: {torch.cuda.get_device_name(0)}", value=True
    )
else:
    use_gpu = st.sidebar.checkbox(
        "Use GPU (Accelerator not available)", value=False, disabled=True
    )

if use_gpu:
    encoding = st.sidebar.selectbox("Encoding", [SupportedEncoding.bfloat16])
else:
    encoding = st.sidebar.selectbox(
        "Encoding",
        [
            SupportedEncoding.q4_k,
            SupportedEncoding.q4_0,
            SupportedEncoding.q6_k,
        ],
    )

max_length = st.sidebar.number_input(
    "Max input and output tokens", 0, 128_000, 12_000
)
max_new_tokens = st.sidebar.number_input("Max output tokens", 0, 24_000, 6000)

hf_file = get_llama_huggingface_file("3.1", encoding)
weights = hf_streamlit_download(hf_file.repo_id, hf_file.filename)

button_state = st.empty()
model_state = st.empty()
if button_state.button("Start Llama3", key=0):
    model_state.info("Starting Llama3...", icon="️⚙️")
    st.session_state["model"] = start_llama3(
        weights,
        encoding,
        max_length,
        max_new_tokens,
        use_gpu,
    )
    model_state.success("Llama3 is ready!", icon="✅")

rag = st.sidebar.checkbox("Activate RAG", value=False)

if rag:
    system_prompt = st.sidebar.text_area(
        "System Prompt",
        value=RAG_SYSTEM_PROMPT,
    )
    n_result = st.sidebar.slider(
        "Number of Top Embedding Search Results", 1, 7, 5
    )
    rag_directory = st.sidebar.text_input(
        "RAG Directory",
        value=Path(__file__).parent.parent / "ragdata",
    )
    filenames = [
        f
        for f in os.listdir(rag_directory)
        if os.path.isfile(os.path.join(rag_directory, f))
    ]
    # Re-cache reading the documents again if there's a change
    collection, embedding_model = load_embed_docs(filenames)
    st.success("RAG data is indexed", icon="✅")
else:
    system_prompt = st.sidebar.text_area(
        "System Prompt",
        value="You are a helpful coding assistant named MAX Llama3.",
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])


disable_chat = True if "model" not in st.session_state else False

if prompt := st.chat_input("Send a message to llama3", disabled=disable_chat):
    messages = [{"role": "system", "content": system_prompt}]
    messages += [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    if rag:
        query_embedding = list(embedding_model.embed(prompt))[0].tolist()
        ret = collection.query(query_embedding, n_results=n_result)
        data = []
        if ret["documents"] is not None and ret["metadatas"] is not None:
            for i, (doc, metadata) in enumerate(
                zip(ret["documents"], ret["metadatas"])
            ):
                data.append(("\n\n".join(doc), metadata[0]["file_name"]))
        messages.append(
            {
                "role": "user",
                "content": RAG_PROMPT.format(query=prompt, data=data),
            }
        )
    else:
        messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="💬"):
        st.markdown(prompt)

    prompt_string = messages_to_llama3_prompt(messages)

    # Sleep short time so prior messages refresh and don't go dark
    time.sleep(0.1)

    with st.chat_message("assistant", avatar="🦙"):
        response = asyncio.run(
            stream_output(st.session_state["model"], prompt_string)
        )

    st.session_state.messages += [
        {"role": "user", "avatar": "💬", "content": prompt},
        {"role": "assistant", "avatar": "🦙", "content": response},
    ]
