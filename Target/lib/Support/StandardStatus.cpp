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
#include "StandardStatus.h"

const char *standardStatusMessages[M_DRIVER_NUM_STANDARD_STATUS_CODES] = {
    /*0*/ "Success",
    /*1*/ "Incompatible version",
    /*2*/ "Unknown error",
    /*3*/ "Argument out of range",
    /*4*/ "Unknown property name",
    /*5*/ "No devices",
    /*6*/ "Bad driver",
    /*7*/ "Bad device",
    /*8*/ "Bad context",
    /*9*/ "Bad memory",
    /*10*/ "Bad queue",
    /*11*/ "Bad event",
    /*12*/ "Bad function",
    /*13*/ "Out of memory",
    /*14*/ "Invalid argument",
    /*15*/ "Incompatible file type",
    /*16*/ "Bad bundle",
    /*17*/ "Execute mode not supported",
    /*18*/ "Unknown symbol",
    /*19*/ "Unsupported",
    /*20*/ "Unimplemented",
};
