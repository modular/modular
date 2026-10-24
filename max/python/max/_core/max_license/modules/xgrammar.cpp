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

// Nanobind bindings for the hack-to-owned xgrammar grammar engine
// (@xgrammar, v0.2.2). Binds only the CPU subset MAX needs.

#include "Support/Log.h"
#include "_core/Graph/Bindings.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"
#include "support/logging.h"
#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nb = nanobind;

// byte-level / tiktoken vocabs (e.g. Kimi) feed each token as raw bytes that
// aren't valid UTF-8, and nanobind's std::string caster only accepts str. So
// accept (bytes | str) and pack raw bytes into std::string, mirroring
// upstream's List[Union[bytes, str]].
static std::vector<std::string> to_byte_vocab(nb::list encoded_vocab) {
  std::vector<std::string> out;
  out.reserve(encoded_vocab.size());
  for (nb::handle item : encoded_vocab) {
    if (nb::isinstance<nb::bytes>(item)) {
      nb::bytes b = nb::borrow<nb::bytes>(item);
      out.emplace_back(static_cast<const char *>(b.c_str()), b.size());
    } else {
      out.push_back(nb::cast<std::string>(item));
    }
  }
  return out;
}

// xgrammar's logger writes every level to stderr, which the log collector
// files as an error; a dropped length bound was landing there thousands of
// times a day. Route the lines through the MAX logger so they carry a level.
// Compiles run on xgrammar's own thread pool, so this never touches Python.
static void forwardXGrammarLog(const std::string &file, int lineno, int level,
                               const std::string &message) {
  switch (level) {
  case XGRAMMAR_LOG_LEVEL_DEBUG:
    MLOG_DEBUG("xgrammar {}:{}: {}", file, lineno, message);
    break;
  case XGRAMMAR_LOG_LEVEL_INFO:
    MLOG_INFO("xgrammar {}:{}: {}", file, lineno, message);
    break;
  default:
    MLOG_WARN("xgrammar {}:{}: {}", file, lineno, message);
    break;
  }
}

void M::Graph::Python::bindXGrammarModule(nb::module_ m) {
  using namespace xgrammar;
  SetLogHandler(&forwardXGrammarLog);

  m.doc() = "Provides bindings for the xgrammar grammar engine (CPU subset).";

  nb::enum_<VocabType>(m, "VocabType")
      .value("RAW", VocabType::RAW)
      .value("BYTE_FALLBACK", VocabType::BYTE_FALLBACK)
      .value("BYTE_LEVEL", VocabType::BYTE_LEVEL);

  nb::class_<TokenizerInfo>(m, "TokenizerInfo")
      .def(
          "__init__",
          [](TokenizerInfo *self, nb::list encoded_vocab, VocabType vocab_type,
             std::optional<int> vocab_size,
             std::optional<std::vector<int32_t>> stop_token_ids,
             bool add_prefix_space,
             std::optional<std::vector<int32_t>> special_token_ids) {
            new (self) TokenizerInfo(to_byte_vocab(encoded_vocab), vocab_type,
                                     vocab_size, stop_token_ids,
                                     add_prefix_space, special_token_ids);
          },
          nb::arg("encoded_vocab"), nb::arg("vocab_type") = VocabType::RAW,
          nb::arg("vocab_size") = nb::none(),
          nb::arg("stop_token_ids") = nb::none(),
          nb::arg("add_prefix_space") = false,
          nb::arg("special_token_ids") = nb::none())
      .def_prop_ro("vocab_type", &TokenizerInfo::GetVocabType)
      .def_prop_ro("vocab_size", &TokenizerInfo::GetVocabSize)
      .def_prop_ro("add_prefix_space", &TokenizerInfo::GetAddPrefixSpace)
      .def_prop_ro("decoded_vocab", &TokenizerInfo::GetDecodedVocab)
      .def_prop_ro("stop_token_ids", &TokenizerInfo::GetStopTokenIds)
      .def_prop_ro("special_token_ids", &TokenizerInfo::GetSpecialTokenIds)
      .def("dump_metadata", &TokenizerInfo::DumpMetadata)
      .def_static(
          "from_vocab_and_metadata",
          [](nb::list encoded_vocab, const std::string &metadata) {
            return TokenizerInfo::FromVocabAndMetadata(
                to_byte_vocab(encoded_vocab), metadata);
          },
          nb::arg("encoded_vocab"), nb::arg("metadata"))
      .def_static("detect_metadata_from_hf",
                  &TokenizerInfo::DetectMetadataFromHF, nb::arg("backend_str"));

  nb::class_<CompiledGrammar>(m, "CompiledGrammar")
      .def_prop_ro("memory_size_bytes", &CompiledGrammar::MemorySizeBytes);

  nb::class_<GrammarCompiler>(m, "GrammarCompiler")
      .def(nb::init<const TokenizerInfo &, int, bool, int64_t>(),
           nb::arg("tokenizer_info"), nb::arg("max_threads") = 8,
           nb::arg("cache_enabled") = true, nb::arg("max_memory_bytes") = -1)
      .def(
          "compile_json_schema",
          [](GrammarCompiler &self, const std::string &schema,
             bool any_whitespace, bool strict_mode, bool require_object_root,
             bool reject_unsupported,
             std::optional<std::pair<std::string, std::string>> separators,
             std::optional<int> max_whitespace_cnt) {
            return self.CompileJSONSchema(
                schema, any_whitespace, std::nullopt, separators, strict_mode,
                max_whitespace_cnt, /*any_order=*/false, require_object_root,
                reject_unsupported);
          },
          nb::arg("schema"), nb::arg("any_whitespace") = true,
          nb::arg("strict_mode") = true, nb::arg("require_object_root") = false,
          nb::arg("reject_unsupported") = false,
          nb::arg("separators") = nb::none(),
          nb::arg("max_whitespace_cnt") = nb::none(),
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "compile_grammar",
          [](GrammarCompiler &self, const std::string &ebnf_str,
             const std::string &root_rule_name) {
            return self.CompileGrammar(ebnf_str, root_rule_name);
          },
          nb::arg("ebnf_str"), nb::arg("root_rule_name") = "root",
          nb::call_guard<nb::gil_scoped_release>())
      .def("compile_builtin_json_grammar",
           &GrammarCompiler::CompileBuiltinJSONGrammar,
           nb::call_guard<nb::gil_scoped_release>())
      .def("compile_structural_tag", &GrammarCompiler::CompileStructuralTag,
           nb::arg("structural_tag_json"),
           nb::call_guard<nb::gil_scoped_release>())
      .def("compile_regex", &GrammarCompiler::CompileRegex, nb::arg("regex"),
           nb::call_guard<nb::gil_scoped_release>())
      .def("clear_cache", &GrammarCompiler::ClearCache)
      .def_prop_ro("cache_size_bytes", &GrammarCompiler::GetCacheSizeBytes)
      .def_prop_ro("cache_limit_bytes", &GrammarCompiler::CacheLimitBytes);

  nb::class_<GrammarMatcher>(m, "GrammarMatcher")
      .def(nb::init<const CompiledGrammar &, std::optional<std::vector<int>>,
                    bool, int>(),
           nb::arg("compiled_grammar"),
           nb::arg("override_stop_tokens") = nb::none(),
           nb::arg("terminate_without_stop_token") = false,
           nb::arg("max_rollback_tokens") = -1)
      .def("accept_token", &GrammarMatcher::AcceptToken, nb::arg("token_id"),
           nb::arg("debug_print") = false,
           nb::call_guard<nb::gil_scoped_release>())
      .def("accept_string", &GrammarMatcher::AcceptString, nb::arg("input_str"),
           nb::arg("debug_print") = false,
           nb::call_guard<nb::gil_scoped_release>())
      .def(
          "fill_next_token_bitmask",
          [](GrammarMatcher &self, nb::ndarray<int32_t, nb::c_contig> bitmask,
             int index, bool debug_print) {
            std::vector<int64_t> shape(bitmask.ndim());
            for (size_t i = 0; i < bitmask.ndim(); ++i)
              shape[i] = static_cast<int64_t>(bitmask.shape(i));
            DLTensor dl{};
            dl.data = bitmask.data();
            dl.device = DLDevice{kDLCPU, 0};
            dl.ndim = static_cast<int32_t>(bitmask.ndim());
            dl.dtype = DLDataType{kDLInt, 32, 1};
            dl.shape = shape.data();
            dl.strides = nullptr;
            dl.byte_offset = 0;
            return self.FillNextTokenBitmask(&dl, index, debug_print);
          },
          nb::arg("bitmask"), nb::arg("index") = 0,
          nb::arg("debug_print") = false,
          nb::call_guard<nb::gil_scoped_release>())
      .def("find_jump_forward_string", &GrammarMatcher::FindJumpForwardString,
           nb::call_guard<nb::gil_scoped_release>())
      .def("rollback", &GrammarMatcher::Rollback, nb::arg("num_tokens") = 1)
      .def("is_terminated", &GrammarMatcher::IsTerminated)
      .def("is_completed", &GrammarMatcher::IsCompleted)
      .def("reset", &GrammarMatcher::Reset)
      .def("fork", &GrammarMatcher::Fork,
           nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("max_rollback_tokens", &GrammarMatcher::GetMaxRollbackTokens)
      .def_prop_ro("stop_token_ids", &GrammarMatcher::GetStopTokenIds);

  m.def("get_bitmask_size", &GetBitmaskSize, nb::arg("vocab_size"),
        "Returns the number of ``int32`` words in a packed bitmask for the "
        "given vocabulary size.");
}
