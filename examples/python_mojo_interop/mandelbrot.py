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

# The Mojo importer module will handle compilation of the Mojo files.
import max._mojo.mojo_importer  # noqa
import os
import sys

sys.path.insert(0, "")

# This is a temporary workaround to prevent conflicts.
os.environ["MOJO_PYTHON_LIBRARY"] = ""

# Importing our Mojo module, defined in the `mandelbrot_mojo.mojo` file.
import mandelbrot_mojo

if __name__ == "__main__":
    # Run the Mandelbrot set calculation on GPU, in Mojo, and get back the ASCII art results.
    ascii_mandelbrot = mandelbrot_mojo.run_mandelbrot(100)
    print(ascii_mandelbrot)
