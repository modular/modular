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
// This file lists NVPTX compiler functions along with their types. This is
// used by CUDACompilationDriver which defines the NVPTX_FUNC macro and then
// includes this to define wrappers, types, and so on, for all the NVPTX
// compiler APIs easily.
//
// To add a new NVPTX compiler API to the driver class, simply add an entry to
// this file in the format:
//
//     NVPTX_FUNC(<API name>, <return type>, <parameter type>...)
//
// Note: For parameters it should be the type only, no parameter names.
//
// Note: If a new API has more than 12 parameters, the macros in
// Support/Preprocessor.h will need to be updated.
//
//===----------------------------------------------------------------------===//

// No macro guards since this is meant to be included multiple times, but make
// sure NVPTX_FUNC is defined.
#ifndef NVPTX_FUNC
#error "CUDACompilationDriver.inc included without defining NVPTX_FUNC"
#endif

NVPTX_FUNC(nvPTXCompilerGetVersion, Result, unsigned int *, unsigned int *)
NVPTX_FUNC(nvPTXCompilerCreate, Result, ::nvPTXCompilerHandle *, size_t, const char *)
NVPTX_FUNC(nvPTXCompilerDestroy, Result, ::nvPTXCompilerHandle *)
NVPTX_FUNC(nvPTXCompilerCompile, Result, ::nvPTXCompilerHandle, int, const char *const *)
NVPTX_FUNC(nvPTXCompilerGetCompiledProgramSize, Result, ::nvPTXCompilerHandle, size_t *)
NVPTX_FUNC(nvPTXCompilerGetCompiledProgram, Result, ::nvPTXCompilerHandle, void *)
NVPTX_FUNC(nvPTXCompilerGetErrorLogSize, Result, ::nvPTXCompilerHandle, size_t *)
NVPTX_FUNC(nvPTXCompilerGetErrorLog, Result, ::nvPTXCompilerHandle, char *)
NVPTX_FUNC(nvPTXCompilerGetInfoLogSize, Result, ::nvPTXCompilerHandle, size_t *)
NVPTX_FUNC(nvPTXCompilerGetInfoLog, Result, ::nvPTXCompilerHandle, char *)
