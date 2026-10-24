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
// Shared test utilities for device graph tests: the vec_add PTX kernel and
// helpers for loading kernels on CUDA and HIP backends.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_UNITTESTS_DRIVER_DEVICEGRAPHTESTUTILS_H
#define MLRT_UNITTESTS_DRIVER_DEVICEGRAPHTESTUTILS_H

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/Driver/DeviceContext/ErrorMacros.h"
#include "tools/cpp/runfiles/runfiles.h"

#include "gtest/gtest.h"

#include "llvm/Support/MemoryBuffer.h"

#include <memory>
#include <string>
#include <string_view>

// From max/kernels/test/gpu/vec_add.ptx
inline constexpr llvm::StringLiteral vecAddPtx = R"PTX(
.version 6.3
.target sm_75
.address_size 64

	// .globl	vec_add

.visible .entry vec_add(
	.param .b64 vec_add_param_0,
	.param .b64 vec_add_param_1,
	.param .b64 vec_add_param_2,
	.param .b64 vec_add_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<28>;
	.reg .f32 	%f<4>;
	.reg .b64 	%rd<12>;

	mov.u32 	%r1, %tid.x;
	cvt.s64.s32 	%rd6, %r1;
	ld.param.u64 	%rd8, [vec_add_param_3];
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %ctaid.x;
	mul.wide.s32 	%rd9, %r2, %r3;
	add.s64 	%rd10, %rd9, %rd6;
	setp.ge.s64 	%p1, %rd10, %rd8;
	@%p1 bra 	$L__BB0_2;
	ld.param.u64 	%rd4, [vec_add_param_0];
	ld.param.u64 	%rd5, [vec_add_param_1];
	ld.param.u64 	%rd7, [vec_add_param_2];
	shl.b64 	%rd11, %rd10, 2;
	add.s64 	%rd1, %rd4, %rd11;
	add.s64 	%rd2, %rd7, %rd11;
	add.s64 	%rd3, %rd5, %rd11;
	ld.u8 	%r4, [%rd3];
	ld.u8 	%r5, [%rd3+1];
	shl.b32 	%r6, %r5, 8;
	or.b32  	%r7, %r6, %r4;
	ld.u8 	%r8, [%rd3+2];
	shl.b32 	%r9, %r8, 16;
	ld.u8 	%r10, [%rd3+3];
	shl.b32 	%r11, %r10, 24;
	or.b32  	%r12, %r11, %r9;
	or.b32  	%r13, %r12, %r7;
	mov.b32 	%f1, %r13;
	ld.u8 	%r14, [%rd2];
	ld.u8 	%r15, [%rd2+1];
	shl.b32 	%r16, %r15, 8;
	or.b32  	%r17, %r16, %r14;
	ld.u8 	%r18, [%rd2+2];
	shl.b32 	%r19, %r18, 16;
	ld.u8 	%r20, [%rd2+3];
	shl.b32 	%r21, %r20, 24;
	or.b32  	%r22, %r21, %r19;
	or.b32  	%r23, %r22, %r17;
	mov.b32 	%f2, %r23;
	add.f32 	%f3, %f2, %f1;
	st.f32 	[%rd1], %f3;
$L__BB0_2:
	ret;
}
)PTX";

inline bazel::tools::cpp::runfiles::Runfiles *getRunfiles() {
  using Runfiles = bazel::tools::cpp::runfiles::Runfiles;
  static std::unique_ptr<Runfiles> runfiles = [] {
    std::string error;
    auto result = std::unique_ptr<Runfiles>(Runfiles::CreateForTest(&error));
    if (!result)
      ADD_FAILURE() << "Failed to create runfiles: " << error;
    return result;
  }();
  return runfiles.get();
}

inline M::ErrorOr<std::string>
loadHIPCodeObject(std::string_view stem, const M::Driver::Device &device) {
  using M::ErrorOr;
  using Runfiles = bazel::tools::cpp::runfiles::Runfiles;

  if (device.api() != M::Driver::Device::hipAPI)
    RETURN_ERROR("loadHIPCodeObject only supports HIP devices");

  DEFINE_OR_RETURN_ERROR(const char *, archName, device.getArchName());
  std::string_view arch(archName);

  if (arch.size() < 6 || arch.substr(0, 6) != "gfx950")
    RETURN_ERROR("Unsupported HIP arch: " + std::string(arch));

  Runfiles *runfiles = getRunfiles();
  if (!runfiles)
    RETURN_ERROR("Runfiles not available");

  std::string runfilePath = "_main/MLRT/unittests/Driver/graph_capture_";
  runfilePath.append(stem.data(), stem.size());
  runfilePath.append("_gfx950.hsaco");
  std::string codeObjectPath = runfiles->Rlocation(runfilePath);
  if (codeObjectPath.empty())
    RETURN_ERROR("Failed to locate runfile: " + runfilePath);

  auto bufferOrErr = llvm::MemoryBuffer::getFile(codeObjectPath);
  if (!bufferOrErr)
    RETURN_ERROR("Failed to open code object: " + codeObjectPath);

  return (*bufferOrErr)->getBuffer().str();
}

inline M::ErrorOr<M::Driver::DeviceFunctionRef>
loadKernelForActiveAPI(M::Driver::DeviceContext &ctx,
                       std::string_view moduleName,
                       std::string_view functionName, std::string_view cudaPtx,
                       std::string_view hipCodeStem) {
  using M::ErrorOr;
  std::string moduleNameStr(moduleName);
  std::string functionNameStr(functionName);

  if (ctx.device().api() == M::Driver::Device::cudaAPI)
    return ctx.loadFunction(moduleNameStr, functionNameStr, cudaPtx);

  if (ctx.device().api() == M::Driver::Device::hipAPI) {
    DEFINE_OR_RETURN_ERROR(std::string, hsacoData,
                           loadHIPCodeObject(hipCodeStem, ctx.device()));
    return ctx.loadFunction(moduleNameStr, functionNameStr, hsacoData);
  }

  RETURN_ERROR("loadKernelForActiveAPI requires CUDA or HIP backend");
}

inline M::ErrorOr<M::Driver::DeviceFunctionRef>
loadVecAdd(M::Driver::DeviceContext &ctx) {
  return loadKernelForActiveAPI(ctx, "vec_add_module", "vec_add", vecAddPtx,
                                "vec_add");
}

#endif // MLRT_UNITTESTS_DRIVER_DEVICEGRAPHTESTUTILS_H
