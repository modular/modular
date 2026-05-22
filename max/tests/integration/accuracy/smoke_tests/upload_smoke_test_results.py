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

# /// script
# dependencies = ["google-cloud-bigquery>=3.0,<4.0"]
# ///

"""Upload smoke test results to BigQuery using a load job (non-streaming).

Uses load_table_from_json rather than the streaming insert API so that rows
are committed to permanent storage immediately, allowing subsequent DML
statements (UPDATE) to touch them without hitting the streaming-buffer
restriction.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone

from google.cloud import bigquery


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload smoke test results to BigQuery."
    )
    parser.add_argument(
        "--path", required=True, help="Path to run_metadata.json"
    )
    parser.add_argument(
        "--project",
        default=os.getenv("GCP_PROJECT_ID", "modular-metrics"),
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv(
            "GCP_RAW_STAGE_DATASET", os.getenv("GCP_DATASET_ID", "")
        ),
    )
    parser.add_argument(
        "--table",
        default=os.getenv(
            "GCP_SERVE_SMOKE_TEST_TABLE", "serve_smoke_test_results"
        ),
    )
    args = parser.parse_args()

    with open(args.path) as f:
        data = json.load(f)

    rows = data.get("modular_metadata", [])
    if not rows:
        print("No rows to upload.")
        return

    now = datetime.now(timezone.utc).isoformat()
    bq_rows = [
        {**row, "id": str(uuid.uuid4()), "created_at": now} for row in rows
    ]

    table_ref = f"{args.project}.{args.dataset}.{args.table}"
    client = bigquery.Client(project=args.project)

    job = client.load_table_from_json(
        bq_rows,
        table_ref,
        job_config=bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            ignore_unknown_values=True,
        ),
    )
    job.result()
    print(f"Uploaded {len(bq_rows)} row(s) to {table_ref}")


if __name__ == "__main__":
    main()
