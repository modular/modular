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

"""Utilities for interacting with HuggingFace Files/Repos."""

from __future__ import annotations

import contextlib
import datetime
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import huggingface_hub
from huggingface_hub import errors as hf_hub_errors
from huggingface_hub import (
    file_exists,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
)
from huggingface_hub.utils import tqdm as hf_tqdm
from requests.exceptions import ConnectionError as RequestsConnectionError
from tqdm.contrib.concurrent import thread_map
from tqdm.std import TqdmDefaultWriteLock
from transformers import AutoConfig

logger = logging.getLogger("max.pipelines")


def get_architectures_from_huggingface_repo(
    model_path: str, trust_remote_code: bool = False
) -> list[str]:
    # Retrieve architecture from model config.
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    return getattr(config, "architectures", [])


@dataclass(frozen=True)
class HuggingFaceFile:
    """A simple object for tracking Hugging Face model metadata. The repo_id will
    frequently be used to load a tokenizer, whereas the filename is used to
    download model weights."""

    repo_id: str
    filename: str
    revision: str | None = None

    def download(self, force_download: bool = False) -> Path:
        """Download the file and return the file path where the data is saved locally."""
        return Path(
            hf_hub_download(
                self.repo_id,
                self.filename,
                revision=self.revision,
                force_download=force_download,
            )
        )

    def size(self) -> int | None:
        url = hf_hub_url(self.repo_id, self.filename, revision=self.revision)
        metadata = get_hf_file_metadata(url)
        return metadata.size

    def exists(self) -> bool:
        return file_exists(self.repo_id, self.filename, revision=self.revision)


class _ThreadingOnlyTqdmLock(TqdmDefaultWriteLock):
    """A version of TqdmDefaultWriteLock that only uses threading locks.

    The tqdm write lock will not be enforced across processes.
    """

    mp_lock = None


@contextlib.contextmanager
def _hf_tqdm_using_threading_only_lock():
    """Use a threading-only lock if there is no existing write lock.

    If a write lock already exists, it is not replaced.  The sole purpose of
    this is to override the default creation of a lock that is problematic in
    this context (as we cannot always ensure proper shutdown of a
    multiprocessing lock, in some cases causing leaks).

    This function exists rather than another hf_tqdm subclass directly
    replacing _lock because Hugging Face internals still use hf_tqdm, and tqdm
    uses class-resident state NOT shared across subclasses, so we have to
    override hf_tqdm directly and cannot use a subclass.
    """
    # N.B.: _lock nonpresence is treated differently than presence with a None
    # value.  Make sure we go down the default path even for None; we only
    # replace the lock if the attribute is not present.  We can't use the
    # public get_lock API for this since that creates the lock we're trying to
    # avoid in the first place.
    if hasattr(hf_tqdm, "_lock"):
        yield
        return
    setattr(hf_tqdm, "_lock", _ThreadingOnlyTqdmLock())
    try:
        yield
    finally:
        delattr(hf_tqdm, "_lock")


def download_weight_files(
    huggingface_model_id: str,
    filenames: list[str],
    revision: Optional[str] = None,
    force_download: bool = False,
    max_workers: int = 8,
) -> list[Path]:
    """Provided a HuggingFace model id, and filenames, download weight files
        and return the list of local paths.

    Args:
        huggingface_model_id:
          The huggingface model identifier, ie. `modularai/llama-3.1`

        filenames:
          A list of file paths relative to the root of the HuggingFace repo.
          If files provided are available locally, download is skipped, and
          the local files are used.

        revision:
          The HuggingFace revision to use. If provided, we check our cache
          directly without needing to go to HuggingFace directly, saving a
          network call.

        force_download:
          A boolean, indicating whether we should force the files to be
          redownloaded, even if they are already available in our local cache,
          or a provided path.

        max_workers:
          The number of worker threads to concurrently download files.

    """
    if not force_download and all(
        os.path.exists(Path(filename)) for filename in filenames
    ):
        logger.info("All files exist locally, skipping download.")
        return [Path(filename) for filename in filenames]

    start_time = datetime.datetime.now()
    logger.info(f"Starting download of model: {huggingface_model_id}")
    with _hf_tqdm_using_threading_only_lock():
        weight_paths = list(
            thread_map(
                lambda filename: Path(
                    hf_hub_download(
                        huggingface_model_id,
                        filename,
                        revision=revision,
                        force_download=force_download,
                    )
                ),
                filenames,
                max_workers=max_workers,
                tqdm_class=hf_tqdm,
            )
        )

    logger.info(
        f"Finished download of model: {huggingface_model_id} in {(datetime.datetime.now() - start_time).total_seconds()} seconds."
    )

    return weight_paths


def repo_exists_with_retry(repo_id: str) -> bool:
    """
    Wrapper around huggingface_hub.repo_exists with retry logic.
    Uses exponential backoff with 25% jitter, starting at 1s and doubling each retry.

    See huggingface_hub.repo_exists for details
    """
    max_attempts = 5
    base_delays = [2**i for i in range(max_attempts)]
    retry_delays_in_seconds = [
        d * (1 + random.uniform(-0.25, 0.25)) for d in base_delays
    ]

    for attempt, delay_in_seconds in enumerate(retry_delays_in_seconds):
        try:
            return huggingface_hub.repo_exists(repo_id)
        except (
            hf_hub_errors.RepositoryNotFoundError,
            hf_hub_errors.GatedRepoError,
            hf_hub_errors.RevisionNotFoundError,
            hf_hub_errors.EntryNotFoundError,
        ) as e:
            # Forward these specific errors to the user
            logger.error(f"Hugging Face repository error: {str(e)}")
            raise
        except (hf_hub_errors.HfHubHTTPError, RequestsConnectionError) as e:
            # Do not retry if Too Many Requests error received
            if e.response.status_code == 429:
                logger.error(e)
                raise

            if attempt == max_attempts - 1:
                logger.error(
                    f"Failed to connect to Hugging Face Hub after {max_attempts} attempts: {str(e)}"
                )
                raise

            logger.warning(
                f"Transient Hugging Face Hub connection error (attempt {attempt + 1}/{max_attempts}): {str(e)}"
            )
            logger.warning(
                f"Retrying Hugging Face connection in {delay_in_seconds} seconds..."
            )
            time.sleep(delay_in_seconds)

    assert False, (
        "This should never be reached due to the raise in the last attempt"
    )
