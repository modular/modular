# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

"""Fetch raw text from a Prometheus metrics endpoint.

A ``requests``-only leaf so ``server_metrics`` and ``gpu_metrics_scraper`` share
one scrape path without importing each other; ``server_metrics`` sits in a
module-load import cycle with ``metrics`` and is unsafe to import first.
"""

from __future__ import annotations

import requests


def fetch_metrics(url: str, *, timeout_s: float = 2.0) -> str:
    """Fetch raw metrics text from a Prometheus endpoint.

    Args:
        url: Prometheus metrics endpoint URL.
        timeout_s: Per-request timeout in seconds.

    Returns:
        Raw Prometheus text format.

    Raises:
        requests.HTTPError: If the response status is not 200.
        requests.RequestException: For network/connection errors.
    """
    response = requests.get(url, timeout=timeout_s)
    if response.status_code != 200:
        raise requests.HTTPError(
            f"Failed to fetch metrics: {response.status_code}",
            response=response,
        )
    return response.text
