//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
// Appended by //GraphCompiler/test/overlay:fixture.
def MO_OverlayTestOp : MO_Op<"overlay_test"> {
  let summary = "Test op added by the overlay fixture.";
  let arguments = (ins MO_Chain:$inChain);
  let results = (outs MO_Chain:$chain);
  let assemblyFormat = "$inChain attr-dict";
  let hasVerifier = 1;
}
