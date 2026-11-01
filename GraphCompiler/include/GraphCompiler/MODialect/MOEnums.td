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
//
// This defines enums in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOENUMS_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOENUMS_TD

include "GraphCompiler/MODialect/MODialect.td"
include "mlir/IR/EnumAttr.td"

class MO_Enum<string name, string summary, list<I32EnumAttrCase> cases> :
  I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::M::MO";
  let genSpecializedAttr = 0;
}

class MO_EnumCase<string sym, int val, string str = sym> :
  I32EnumAttrCase<sym, val, str>;

def MO_CoordinateTransformMode : MO_Enum<
  "CoordinateTransformMode", "Valid coordinate transformations for `mo.resize`", [
    I32EnumAttrCase<"HalfPixel", 0, "half_pixel">,
    I32EnumAttrCase<"AlignCorners", 1, "align_corners">,
    I32EnumAttrCase<"Asymmetric", 2, "asymmetric">,
    I32EnumAttrCase<"HalfPixel1D", 3, "half_pixel_1D">
  ]
>;

def MO_CoordinateTransformModeAttr : EnumAttr<MODialect, MO_CoordinateTransformMode, "coord_trans_mode">
{
  let summary = "coordinate transformations";
  let description = [{
    This attribute is used by `mo.resize`.
  }];

  let cppNamespace = "::M::MO";
}

def MO_IOKindEnum : MO_Enum<"IOKind", "IO type of a mojo kernel operand", [
  MO_EnumCase<"Unknown", 32>, // Can't do -1.
  MO_EnumCase<"Output", 0>,
  MO_EnumCase<"Input", 1>,
  MO_EnumCase<"FusedInput", 2>,
  MO_EnumCase<"FusedOutput", 3>,
  MO_EnumCase<"FusedComputeOutput", 31>
]>;

// Prints as `#mo.io<Input>` rather than the case's integer.
def MO_IOKind : EnumAttr<MODialect, MO_IOKindEnum, "io">;

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOENUMS_TD
