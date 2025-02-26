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
import threading
import time
from functools import wraps
from pathlib import Path
from typing import List

import chromadb
import streamlit as st
from chromadb.config import Settings
from fastembed import TextEmbedding
from gguf import Union
from huggingface_hub import hf_hub_download, snapshot_download
from llama_index.core import SimpleDirectoryReader
from max.pipelines import TokenGenerator
from streamlit.runtime.scriptrunner import (
    add_script_run_ctx,
    get_script_run_ctx,
)
from tqdm.auto import tqdm

RAG_SYSTEM_PROMPT = """You are a helpful document search assistant.
Your task is to find an answer to user's query about their given documentation.
DO NOT HALLUCINATE."""

RAG_PROMPT = """Answer the users query: {query} using the provided context: {data}.
If you don't have an answer say 'I don't know!'
Make sure to include the filename of any document you use to answer the query.

## GO"""


def menu():
    st.sidebar.page_link("home.py", label="️Home", icon="⚡️")
    st.sidebar.page_link("pages/llama3_1.py", label="Llama3.1", icon="🦙")
    st.sidebar.page_link("pages/bert.py", label="Bert", icon="👓")
    st.sidebar.page_link("pages/yolo.py", label="YOLO", icon="🔍")
    st.sidebar.page_link(
        "pages/stable-diffusion.py", label="Stable Diffusion 1.5", icon="🎨"
    )


def modular_cache_dir() -> str:
    cache_folder = os.getenv("XDG_CACHE_PATH", str(Path.home() / ".cache"))
    modular_dir = os.path.join(cache_folder, "modular")
    os.makedirs(modular_dir, exist_ok=True)
    return modular_dir


def format_time(seconds):
    """Return a pretty format based on how much time is left."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"


tqdm_patched = False


def hf_streamlit_download(repo_id: str, filename: str = "") -> str:
    """Patch tqdm to update st.progress bars, uses `hf_hub_download` if
    a `repo_id` and `filename` is provided, otherwise uses `snapshot_download`.
    """
    global tqdm_patched
    if not tqdm_patched:
        tqdm_patched = True
        patch_tqdm()

    if filename:
        return hf_hub_download(repo_id, filename)

    return snapshot_download(repo_id)


# TODO: raise patch to `huggingface_hub` to see if they'd be interested in this
# as a class that inherits from tqdm, which can be installed with
# huggingface_hub[streamlit]
def patch_tqdm():
    """Monkey patches tqdm to update st.progress bars. `hf_hub_download` and
    `snapshot_download` use tqdm, so this is a way to hook into the progress
    of each file download and display it on the GUI. We need to pass the
    script context so other threads can communicate with the main thread,
    when interacting with streamlit elements."""
    progress_bars = {}
    ctx = get_script_run_ctx()

    def patch_update(original):
        @wraps(original)
        def wrapper(self, n=1):
            # Return early if tqdm instance is describing how many files
            # will be downloaded.
            if self.desc.startswith("Fetching"):
                return

            if self.n is not None and self.total is not None:
                # Convert everything to MB
                downloaded = self.n / 1024 / 1024
                total = self.total / 1024 / 1024
                speed = (
                    self.format_dict["rate"] / 1024 / 1024
                    if self.format_dict["rate"]
                    else 0.0
                )
                if speed != 0:
                    time_left = format_time((total - downloaded) / speed)
                else:
                    time_left = "N/A"
                # Only create/update progress bar if the download is in progress to skip small files
                if time_left != "N/A":
                    add_script_run_ctx(threading.currentThread(), ctx)
                    if self.pos not in progress_bars:
                        progress_bars[self.pos] = st.empty()
                    progress_bar = progress_bars[self.pos]
                    status = (
                        f"{self.desc}: {int(downloaded)}/{int(total)} MB Speed:"
                        f" {speed:.2f} MB/s Remaining: {time_left}"
                    )
                    progress_bar.progress(downloaded / total, status)
            return original(self, n)

        return wrapper

    def patch_del(original):
        @wraps(original)
        def wrapper(self, *args, **kwargs):
            if self.pos in progress_bars:
                progress_bars[self.pos].empty()
                del progress_bars[self.pos]

            return original(self, *args, **kwargs)

        return wrapper

    tqdm.update = patch_update(tqdm.update)
    tqdm.__del__ = patch_del(tqdm.__del__)


@st.cache_resource(show_spinner=False)
def load_embed_docs(docs_filenames: List[str]):
    """Loads documents from `./ragdir` and embeds them to chromadb
    using a text embedding model.
    """
    with st.spinner("Loading RAG data..."):
        docs = SimpleDirectoryReader("./ragdata").load_data()
        client = chromadb.Client(Settings(anonymized_telemetry=False))
        collection = client.get_or_create_collection(
            "max-rag-example", metadata={"hnsw:space": "cosine"}
        )
        embedding_model = TextEmbedding()

        for i, doc in enumerate(docs):
            embedding = list(embedding_model.embed(doc.text))[0].tolist()
            collection.upsert(
                documents=doc.text,
                embeddings=embedding,
                ids=[str(i)],
                metadatas=[doc.metadata],
            )

        return collection, embedding_model


# Generate metrics for streamlit
class TextGenerationMetrics:
    """Metrics capturing and reporting for a text generation pipeline."""

    prompt_size: int
    output_size: int
    startup_time: Union[float, str]
    time_to_first_token: Union[float, str]
    prompt_eval_throughput: Union[float, str]
    eval_throughput: Union[float, str]

    _start_time: float
    _signposts: dict[str, float]

    def __init__(self):
        self.signposts = {}
        self.prompt_size = 0
        self.output_size = 0
        self.start_time = time.time()

    def signpost(self, name: str):
        """Measure the current time and tag it with a name for later reporting."""
        self.signposts[name] = time.time()

    def new_token(self):
        """Report that a new token has been generated."""
        self.output_size += 1

    def calculate_results(self):
        end_generation = time.time()
        begin_generation = self.signposts.get("begin_generation")
        if begin_generation:
            self.startup_time = (
                self.signposts["begin_generation"] - self.start_time
            ) * 1000.0
        else:
            self.startup_time = "n/a"

        first_token = self.signposts.get("first_token")
        if first_token and begin_generation:
            self.time_to_first_token = (
                self.signposts["first_token"]
                - self.signposts["begin_generation"]
            ) * 1000.0
        else:
            self.time_to_first_token = "n/a"

        st.sidebar.metric(
            "Input/Output Tokens",
            value=f"{self.prompt_size}/{self.output_size}",
        )
        st.sidebar.metric(
            "Time to first token", value=f"{self.time_to_first_token:.2f} ms"
        )

        if first_token and begin_generation:
            generation_time = end_generation - self.signposts["first_token"]
            assert isinstance(self.time_to_first_token, float)
            self.prompt_eval_throughput = self.prompt_size / (
                self.time_to_first_token / 1000.0
            )
            self.eval_throughput = (self.output_size - 1) / generation_time
            st.sidebar.metric(
                "Prompt eval throughput (context-encoding):",
                value=f"{self.prompt_eval_throughput:.2f} tokens/s",
            )
            st.sidebar.metric(
                "Eval throughput (token-generation):",
                value=f"{self.eval_throughput:.2f} tokens/s",
            )


async def stream_output(model: TokenGenerator, prompt: str) -> str:
    metrics = TextGenerationMetrics()
    context = model.new_context(prompt)
    prompt_size = context.current_length

    response_display = st.empty()
    response_str = ""

    if metrics:
        metrics.prompt_size = prompt_size
        metrics.signpost("begin_generation")

    is_first_token = True
    request_id = str(id(prompt))
    while True:
        response = model.next_token({request_id: context}, num_steps=1)[0]
        if request_id not in response:
            break
        response_str += response[request_id]
        response_display.markdown(response_str)
        if metrics:
            if is_first_token:
                is_first_token = False
                metrics.signpost("first_token")
            metrics.new_token()
    if metrics:
        metrics.signpost("end_generation")

    metrics.calculate_results()
    return response_str
