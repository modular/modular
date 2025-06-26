# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Modified version of GGUFReader that runs quicker for our purposes."""

import contextlib
from typing import Any

import gguf  # type: ignore
import numpy as np
import numpy.typing as npt


class TokenSkippingGGUFReader(gguf.GGUFReader):
    """GGUF file reader that skips over some tokenization metadata.

    Tokenization metadata makes up a substantial portion of the metadata in
    many GGUF files, and fully parsing it takes a noticeable amount of time
    (several seconds).  When we are reading a GGUF file for its tensors only
    and don't need the tokenization metadata, skipping over the tokenization
    metadata saves a lot of time, allowing a full parse to take <100ms rather
    than several seconds.
    """

    def _build_fields(self, offs: int, count: int) -> int:
        for _ in range(count):
            orig_offs = offs
            kv_klen, kv_kdata = self._get_str(offs)
            key = str(bytes(kv_kdata), encoding="utf-8")
            offs += int(kv_klen.nbytes + kv_kdata.nbytes)
            raw_kv_type = self._get(offs, np.uint32)
            offs += int(raw_kv_type.nbytes)
            if key == "tokenizer.ggml.tokens" or key == "tokenizer.ggml.merges":
                offs += self._skip_string_array(offs)
                continue
            if key == "tokenizer.ggml.token_type":
                offs += self._skip_int32_array(offs)
                continue
            parts: list[npt.NDArray[Any]] = [kv_klen, kv_kdata, raw_kv_type]
            idxs_offs = len(parts)
            field_size, field_parts, field_idxs, field_types = (
                self._get_field_parts(offs, raw_kv_type[0])
            )
            parts += field_parts
            self._push_field(
                gguf.ReaderField(
                    orig_offs,
                    key,
                    parts,
                    [idx + idxs_offs for idx in field_idxs],
                    field_types,
                ),
                skip_sum=True,
            )
            offs += field_size
        return offs

    def _skip_string_array(self, offs: int) -> int:
        orig_offs = offs
        el_type = self._get(offs, np.uint32)
        if gguf.GGUFValueType(el_type[0]) != gguf.GGUFValueType.STRING:
            raise ValueError(
                "_skip_string_array can only be used to skip over string arrays"
            )
        offs += el_type.nbytes
        count = self._get(offs, np.uint64)
        offs += count.nbytes
        # The performance of the following loop is a bottleneck, pinpointed by
        # a profiler as a major contributor to execution time, so we take some
        # care to set things up so that the body of the main loop can do its
        # work as efficiently as possible.
        #
        # We need to be able to quickly access qwords at an arbitrary unaligned
        # byte offset.  There is no natural way to do that, but it's easy to
        # access qwords at qword indices from a starting byte offset.  A qword
        # is 8 bytes long, so we create a "qword view" for each possible byte
        # offset.  Then to access at a particular byte offset, we can pick out
        # the existing pre-created qword view based on the remainder and access
        # it at the qword index based on the quotient.
        #
        # We use memoryview on the underlying data instead of going through
        # NumPy since NumPy has too many layers of abstraction, slowing this
        # loop down.  memoryview is fairly raw and has a lower cost, but we do
        # need to manage its lifetime (and every derived memoryview value,
        # including slices) with a context manager to avoid locking the
        # underlying mmap's buffer.
        with contextlib.ExitStack() as exit_stack:
            data_view = exit_stack.enter_context(memoryview(self.data._mmap))
            qword_views = [
                exit_stack.enter_context(
                    exit_stack.enter_context(
                        data_view[
                            qword_byte_offset : qword_byte_offset
                            + ((len(data_view) - qword_byte_offset) & ~7)
                        ]
                    ).cast("Q")
                )
                for qword_byte_offset in range(8)
            ]
            for i in range(int(count[0])):
                str_length = qword_views[offs % 8][offs // 8]
                offs += str_length + 8
            return offs - orig_offs

    def _skip_int32_array(self, offs: int) -> int:
        orig_offs = offs
        el_type = self._get(offs, np.uint32)
        if gguf.GGUFValueType(el_type[0]) != gguf.GGUFValueType.INT32:
            raise ValueError(
                "_skip_int32_array can only be used to skip over int32 arrays"
            )
        offs += el_type.nbytes
        count = self._get(offs, np.uint64)
        offs += count.nbytes
        offs += int(count[0]) * 4
        return offs - orig_offs
