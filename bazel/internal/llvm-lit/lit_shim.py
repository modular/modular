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

from lit.main import main

if __name__ == "__main__":
    xml_output = os.environ["XML_OUTPUT_FILE"]
    lit_opts = os.environ.get("LIT_OPTS", "")
    lit_opts = f"{lit_opts} --xunit-xml-output {xml_output}".strip()
    os.environ["LIT_OPTS"] = lit_opts
    main()
