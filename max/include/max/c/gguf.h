//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef MAX_C_GGUF_H
#define MAX_C_GGUF_H

#include "max/c/symbol_export.h"
#include "max/c/types.h"

#ifndef __cplusplus
#include <stdbool.h>
#endif

/// Loads and parses one or more GGUF files onto a device.
///
/// The tensors from every file are flattened into a single namespace keyed by
/// tensor name. It is an error for the same tensor name to appear in more than
/// one file.
///
/// The data is loaded onto `device`. To read tensor data on the host (for
/// example, to build a weights registry with `M_newWeightsRegistryFromGGUF()`),
/// pass a host device created with `M_newDevice(M_HOST, ...)`. The runtime
/// copies the weights to an accelerator when the model is initialized.
///
/// Block-quantized tensors are exposed as `uint8` data whose last dimension is
/// scaled from elements to quantization blocks, so their shape and byte count
/// describe the stored bytes rather than the logical element shape. Use
/// `M_getGGUFQuantization()` to tell the two apart.
///
/// @param paths An array of null-terminated file paths to GGUF files.
/// @param numPaths The number of entries in `paths`.
/// @param device The device to load the tensor data onto.
/// @param status The status object for reporting errors.
///
/// @returns A pointer to the loaded GGUF. You are responsible for the memory
/// associated with the pointer returned, which can be deallocated by calling
/// `M_freeGGUF()`. Returns `NULL` if loading fails, with an error message in
/// `status`.
MODULAR_API_EXPORT M_GGUF *M_loadGGUF(const char **paths, size_t numPaths,
                                      M_Device *device, M_Status *status);

/// Returns the number of tensors across all loaded GGUF files.
///
/// @param gguf The loaded GGUF.
MODULAR_API_EXPORT size_t M_getGGUFCount(const M_GGUF *gguf);

/// Returns the name of the tensor at the given index.
///
/// Indices are stable for the lifetime of the `M_GGUF` object and range from
/// `0` to `M_getGGUFCount() - 1`. The returned string is owned by the `M_GGUF`
/// object and must not be freed by the caller.
///
/// @param gguf The loaded GGUF.
/// @param index The tensor index.
///
/// @returns The null-terminated tensor name, or `NULL` if `index` is out of
/// range.
MODULAR_API_EXPORT const char *M_getGGUFName(const M_GGUF *gguf, size_t index);

/// Returns a pointer to the raw data for the named tensor.
///
/// The pointer is **borrowed**: it is valid for the lifetime of the `M_GGUF`
/// object and must not be freed by the caller. When the GGUF files were loaded
/// onto a host device, this is a host-accessible pointer.
///
/// @param gguf The loaded GGUF.
/// @param name The null-terminated tensor name.
/// @param status The status object for reporting errors.
///
/// @returns A pointer to the tensor data, or `NULL` if `name` is not present,
/// with an error message in `status`.
MODULAR_API_EXPORT const void *
M_getGGUFData(const M_GGUF *gguf, const char *name, M_Status *status);

/// Returns the size in bytes of the named tensor's data.
///
/// @param gguf The loaded GGUF.
/// @param name The null-terminated tensor name.
/// @param status The status object for reporting errors.
///
/// @returns The size in bytes, or `0` if `name` is not present, with an error
/// message in `status`.
MODULAR_API_EXPORT size_t M_getGGUFNumBytes(const M_GGUF *gguf,
                                            const char *name, M_Status *status);

/// Returns the element data type of the named tensor.
///
/// Block-quantized tensors have type `M_UINT8`: their data is a sequence of
/// quantization blocks.
///
/// @param gguf The loaded GGUF.
/// @param name The null-terminated tensor name.
/// @param status The status object for reporting errors.
///
/// @returns The tensor's `M_Dtype`, or `M_UNKNOWN` if `name` is not present or
/// its quantization scheme is not supported, with an error message in
/// `status`.
MODULAR_API_EXPORT M_Dtype M_getGGUFDtype(const M_GGUF *gguf, const char *name,
                                          M_Status *status);

/// Returns the number of dimensions (rank) of the named tensor's stored data.
///
/// @param gguf The loaded GGUF.
/// @param name The null-terminated tensor name.
/// @param status The status object for reporting errors.
///
/// @returns The rank, or `0` if `name` is not present, with an error message
/// in `status`. Note that a scalar tensor also has rank `0`; check `status` to
/// disambiguate.
MODULAR_API_EXPORT size_t M_getGGUFRank(const M_GGUF *gguf, const char *name,
                                        M_Status *status);

/// Writes the shape of the named tensor's stored data into a caller-provided
/// buffer.
///
/// For block-quantized tensors the last dimension is in quantization blocks,
/// matching `M_getGGUFDtype()` and `M_getGGUFNumBytes()`. The caller must
/// provide a buffer with space for at least `M_getGGUFRank()` entries.
///
/// @param gguf The loaded GGUF.
/// @param name The null-terminated tensor name.
/// @param shapeOut A buffer that receives the dimension sizes.
/// @param status The status object for reporting errors.
MODULAR_API_EXPORT void M_getGGUFShape(const M_GGUF *gguf, const char *name,
                                       int64_t *shapeOut, M_Status *status);

/// Returns the quantization scheme of the named tensor.
///
/// @param gguf The loaded GGUF.
/// @param name The null-terminated tensor name.
///
/// @returns The null-terminated scheme name, such as `"Q4_K"`, or `NULL` when
/// the tensor is not block-quantized or `name` is not present.
MODULAR_API_EXPORT const char *M_getGGUFQuantization(const M_GGUF *gguf,
                                                     const char *name);

/// Returns the GGUF format version of the files.
///
/// @param gguf The loaded GGUF.
MODULAR_API_EXPORT uint32_t M_getGGUFVersion(const M_GGUF *gguf);

/// Returns the alignment of the tensor data section of the files.
///
/// @param gguf The loaded GGUF.
MODULAR_API_EXPORT size_t M_getGGUFAlignment(const M_GGUF *gguf);

/// Builds a weights registry mapping every loaded tensor name to its data.
///
/// This is a convenience over `M_newWeightsRegistry()` for the common case of
/// initializing a model directly from GGUF files. The tensor names in the files
/// must match the external weight names (`constant_external`) in the compiled
/// graph; this function does not perform any name translation.
///
/// The returned registry **borrows** the tensor data from `gguf`, which must
/// outlive both the registry and any model initialized from it.
///
/// @param gguf The loaded GGUF.
/// @param status The status object for reporting errors.
///
/// @returns A pointer to the weights registry, to be freed with
/// `M_freeWeightsRegistry()`. Returns `NULL` on failure, with an error message
/// in `status`.
MODULAR_API_EXPORT M_WeightsRegistry *
M_newWeightsRegistryFromGGUF(const M_GGUF *gguf, M_Status *status);

/// Deallocates the memory for the loaded GGUF. No-op if `gguf` is `NULL`.
///
/// Any pointers returned by `M_getGGUFData()` and any weights registry created
/// by `M_newWeightsRegistryFromGGUF()` are invalidated.
///
/// @param gguf The GGUF to deallocate.
MODULAR_API_EXPORT void M_freeGGUF(M_GGUF *gguf);

/// Creates a GGUF writer.
///
/// Metadata and tensors are recorded in the order they are added, which is the
/// order they appear in the written file. Tensor data is copied into the
/// writer, so the caller's buffer may be released before `M_writeGGUF()`.
///
/// @param alignment The alignment of the tensor data section; `0` selects the
/// format's default of 32 bytes.
/// @param status The status object for reporting errors.
///
/// @returns A pointer to the writer, to be freed with `M_freeGGUFWriter()`.
/// Returns `NULL` if the alignment is not a power of two, with an error
/// message in `status`.
MODULAR_API_EXPORT M_GGUFWriter *M_newGGUFWriter(size_t alignment,
                                                 M_Status *status);

/// Adds or replaces a string-valued metadata entry.
MODULAR_API_EXPORT void M_setGGUFMetadataString(M_GGUFWriter *writer,
                                                const char *key,
                                                const char *value);

/// Adds or replaces a boolean-valued metadata entry.
MODULAR_API_EXPORT void M_setGGUFMetadataBool(M_GGUFWriter *writer,
                                              const char *key, bool value);

/// Adds or replaces a signed integer metadata entry, written as an `INT64`.
MODULAR_API_EXPORT void M_setGGUFMetadataInt(M_GGUFWriter *writer,
                                             const char *key, int64_t value);

/// Adds or replaces an unsigned integer metadata entry, written as a `UINT64`.
MODULAR_API_EXPORT void M_setGGUFMetadataUInt(M_GGUFWriter *writer,
                                              const char *key, uint64_t value);

/// Adds or replaces a floating point metadata entry, written as a `FLOAT64`.
MODULAR_API_EXPORT void M_setGGUFMetadataFloat(M_GGUFWriter *writer,
                                               const char *key, double value);

/// Adds a tensor.
///
/// `tensorType` is the GGML type of `data`, and `shape` is the logical shape,
/// widest dimension first. For block-quantized types the last dimension counts
/// elements, so keep it a whole number of quantization blocks.
///
/// @param writer The writer.
/// @param name The null-terminated tensor name.
/// @param tensorType The GGML tensor type id.
/// @param shape The tensor's logical shape.
/// @param rank The number of entries in `shape`.
/// @param data The tensor data.
/// @param numBytes The size of `data` in bytes.
/// @param status The status object for reporting errors.
MODULAR_API_EXPORT void M_addGGUFTensor(M_GGUFWriter *writer, const char *name,
                                        uint32_t tensorType,
                                        const int64_t *shape, size_t rank,
                                        const void *data, size_t numBytes,
                                        M_Status *status);

/// Writes the file to `path`, replacing it if it exists.
///
/// @param writer The writer.
/// @param path The null-terminated destination path.
/// @param status The status object for reporting errors.
MODULAR_API_EXPORT void M_writeGGUF(const M_GGUFWriter *writer,
                                    const char *path, M_Status *status);

/// Deallocates a writer. No-op if `writer` is `NULL`.
MODULAR_API_EXPORT void M_freeGGUFWriter(M_GGUFWriter *writer);

#endif // MAX_C_GGUF_H
