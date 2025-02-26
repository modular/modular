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

import subprocess

from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello, magic!"}


@app.get("/zero")
def zero():
    try:
        p = subprocess.Popen(
            ["magic", "run", "mojo", "local/zero.mojo"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while True:
            output = p.stdout.readline()
            if output == "" and p.poll() is not None:
                raise HTTPException(
                    status_code=500, detail="Failed to produce zero"
                )

            return {"message": f"answer is {output}"}

    except subprocess.SubprocessError:
        raise HTTPException(
            status_code=500, detail="Failed to execute subprocess"
        )
