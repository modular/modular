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

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "_core/Graph/Bindings.h"
#include "nanobind/nanobind.h"

// The upstream NIXL library (libNIXL.so, libNIXL_common.so, etc.) only ships
// prebuilt binaries for linux-x86_64.  Methods like remDesc(), addDesc(),
// getIndex(), trim(), print(), operator==, and all nixlAgent / nixlAgentConfig
// entry points are *declared* in the public headers but *defined* only in those
// libraries.  Linking them on aarch64 or macOS produces undefined-symbol errors
// that prevent _core.so from loading at all, breaking every _core consumer on
// those platforms.
//
// Guard the entire NIXL binding body behind linux-x86_64.  On every other
// platform, bindNIXLModule() registers an empty but importable submodule so
// that `from max._core import nixl` in transfer_engine.py does not raise an
// ImportError at module-load time (the actual nixl Agent / descriptor APIs are
// never reachable on non-linux-x86 in production).
#if defined(__linux__) && defined(__x86_64__)

#include "nanobind/ndarray.h"
#include "nanobind/operators.h"
#include "nanobind/stl/map.h"      // IWYU pragma: keep (std::map typecaster)
#include "nanobind/stl/optional.h" // IWYU pragma: keep (std::optional typecaster)
#include "nanobind/stl/pair.h"     // IWYU pragma: keep (std::pair typecaster)
#include "nanobind/stl/string.h"   // IWYU pragma: keep (std::string typecaster)
#include "nanobind/stl/string_view.h" // IWYU pragma: keep (std::string_view typecaster)
#include "nanobind/stl/tuple.h" // IWYU pragma: keep (std::tuple typecaster)
#include "nanobind/stl/unordered_map.h" // IWYU pragma: keep (std::unordered_map typecaster)
#include "nanobind/stl/variant.h" // IWYU pragma: keep (std::variant typecaster)
#include "nanobind/stl/vector.h"  // IWYU pragma: keep (std::vector typecaster)

#include "nixl.h"
#include "nixl_descriptors.h"
#include "nixl_types.h"

#include <chrono>
#include <map>
#include <tuple>
#include <variant>

namespace nb = nanobind;
using namespace nb::literals;

using nixl_py_notifs_t = std::map<std::string, std::vector<nb::bytes>>;

struct nixlNotPostedError : std::runtime_error {
  nixlNotPostedError(const std::string &what) : runtime_error(what) {}
};

struct nixlInvalidParamError : std::runtime_error {
  nixlInvalidParamError(const std::string &what) : runtime_error(what) {}
};

struct nixlBackendError : std::runtime_error {
  nixlBackendError(const std::string &what) : runtime_error(what) {}
};

struct nixlNotFoundError : std::runtime_error {
  nixlNotFoundError(const std::string &what) : runtime_error(what) {}
};

struct nixlMismatchError : std::runtime_error {
  nixlMismatchError(const std::string &what) : runtime_error(what) {}
};

struct nixlNotAllowedError : std::runtime_error {
  nixlNotAllowedError(const std::string &what) : runtime_error(what) {}
};

struct nixlRepostActiveError : std::runtime_error {
  nixlRepostActiveError(const std::string &what) : runtime_error(what) {}
};

struct nixlNotSupportedError : std::runtime_error {
  nixlNotSupportedError(const std::string &what) : runtime_error(what) {}
};

struct nixlUnknownError : std::runtime_error {
  nixlUnknownError(const std::string &what) : runtime_error(what) {}
};

struct nixlRemoteDisconnectError : std::runtime_error {
  nixlRemoteDisconnectError(const std::string &what) : runtime_error(what) {}
};

struct nixlCanceledError : std::runtime_error {
  nixlCanceledError(const std::string &what) : runtime_error(what) {}
};

struct nixlNoTelemetryError : std::runtime_error {
  nixlNoTelemetryError(const std::string &what) : runtime_error(what) {}
};

void throw_nixl_exception(const nixl_status_t status) {
  switch (status) {
  case nixl_status_t::NIXL_IN_PROG:
  case nixl_status_t::NIXL_SUCCESS:
    // Not errors
    return;
  case nixl_status_t::NIXL_ERR_NOT_POSTED:
    throw nixlNotPostedError(nixlEnumStrings::statusStr(status));
  case nixl_status_t::NIXL_ERR_INVALID_PARAM:
    throw nixlInvalidParamError(nixlEnumStrings::statusStr(status));
  case nixl_status_t::NIXL_ERR_BACKEND:
    throw nixlBackendError(nixlEnumStrings::statusStr(status));
  case nixl_status_t::NIXL_ERR_NOT_FOUND:
    throw nixlNotFoundError(nixlEnumStrings::statusStr(status));
  case nixl_status_t::NIXL_ERR_MISMATCH:
    throw nixlMismatchError(nixlEnumStrings::statusStr(status));
  case nixl_status_t::NIXL_ERR_NOT_ALLOWED:
    throw nixlNotAllowedError(nixlEnumStrings::statusStr(status));
  case nixl_status_t::NIXL_ERR_REPOST_ACTIVE:
    throw nixlRepostActiveError(nixlEnumStrings::statusStr(status));
  case nixl_status_t::NIXL_ERR_UNKNOWN:
    throw nixlUnknownError(nixlEnumStrings::statusStr(status));
  case nixl_status_t::NIXL_ERR_NOT_SUPPORTED:
    throw nixlNotSupportedError(nixlEnumStrings::statusStr(status));
  case nixl_status_t::NIXL_ERR_REMOTE_DISCONNECT:
    throw nixlRemoteDisconnectError(nixlEnumStrings::statusStr(status));
  case nixl_status_t::NIXL_ERR_CANCELED:
    throw nixlCanceledError(nixlEnumStrings::statusStr(status));
  case nixl_status_t::NIXL_ERR_NO_TELEMETRY:
    throw nixlNoTelemetryError(nixlEnumStrings::statusStr(status));
  }
}

using RegistrationTuple = std::tuple<uintptr_t, size_t, uint64_t, std::string>;
using TransferTuple = std::tuple<uintptr_t, size_t, uint64_t>;

struct nixlPyXferTelemetry {
  long long startTime;
  long long postDuration;
  long long xferDuration;
  size_t totalBytes;
  size_t descCount;
};

nixlPyXferTelemetry pyXferTelemetryFromCpp(const nixl_xfer_telem_t &telemetry) {
  return {
      std::chrono::duration_cast<chrono_period_us_t>(
          telemetry.startTime.time_since_epoch())
          .count(),
      telemetry.postDuration.count(),
      telemetry.xferDuration.count(),
      telemetry.totalBytes,
      telemetry.descCount,
  };
}

nixlBlobDesc tupleToElt(RegistrationTuple tup) {
  auto &[addr, len, devID, metaInfo] = tup;
  return {addr, len, devID, std::move(metaInfo)};
}

nixlBasicDesc tupleToElt(TransferTuple tup) {
  auto &[addr, len, devID] = tup;
  return {addr, len, devID};
}

RegistrationTuple eltToTuple(const nixlBlobDesc &elt) {
  return {elt.addr, elt.len, elt.devId, elt.metaInfo};
}

TransferTuple eltToTuple(const nixlBasicDesc &elt) {
  return {elt.addr, elt.len, elt.devId};
}

/// Converts a Python object into a specified element, either
/// by interpreting it as a DLPack object or a tuple. Throws
/// a type error if the object is neither.
template <typename Elt>
Elt objectToElt(nb::handle elt) {
  using Tup = decltype(eltToTuple(Elt()));

  if (nb::ndarray<> arr; nb::try_cast(elt, arr)) {
    return {reinterpret_cast<uintptr_t>(arr.data()), arr.nbytes(),
            static_cast<uint64_t>(arr.device_id())};
  } else if (Tup tup; nb::try_cast(elt, tup)) {
    return tupleToElt(std::move(tup));
  } else {
    throw nb::type_error(
        "Unrecognized type passed to descriptor list, expected a "
        "dlpack object or a tuple");
  }
}

template <typename Elt>
void bindDlist(nb::class_<nixlDescList<Elt>> &cls) {
  using DList = nixlDescList<Elt>;
  using Tup = decltype(eltToTuple(Elt()));
  using DLPackOrTuple = std::variant<nb::ndarray<>, Tup>;
  cls.def(nb::init<nixl_mem_t, int>(), "type"_a, "init_size"_a = 0, R"nbdoc(
        Constructs an empty descriptor list.

        Args:
          type: The type of memory each element describes.
          init_size: Initial capacity of the list.
      )nbdoc")
      .def(
          "__init__",
          [](DList *out, nixl_mem_t mem,
             nb::typed<nb::sequence, DLPackOrTuple> descs) {
            new (out) DList(mem);
            for (auto elt : descs) {
              out->addDesc(objectToElt<Elt>(elt));
            }
          },
          "type"_a, "descs"_a, R"nbdoc(
        Constructs a descriptor list with given values.

        Args:
          type: The type of memory each element describes.
          descs: A list of descriptors, each describing a section of memory.
                 Each element is either a tuple or a dlpack object.
      )nbdoc")
      .def(
          "append",
          [](DList &self, const nb::typed<nb::object, DLPackOrTuple> &desc) {
            self.addDesc(objectToElt<Elt>(desc));
          },
          "desc"_a, R"nbdoc(
        Adds a descriptor to the list.

        Args:
          desc: A descriptor describing a section of memory.
                Either a tuple or a dlpack object.
      )nbdoc");
}

void M::Graph::Python::bindNIXLModule(nb::module_ m) {
  // TODO: each nixl class and/or function can be documented in place
  m.doc() = "nanobind NIXL plugin: Implements NIXL descriptors and lists, as "
            "well as bindings of NIXL CPP APIs.";

  m.attr("INIT_AGENT") = NIXL_INIT_AGENT;

  m.attr("DEFAULT_COMM_PORT") = default_comm_port;

  // cast types
  nb::enum_<nixl_thread_sync_t>(m, "ThreadSyncMode")
      .value("NONE", nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE)
      .value("STRICT", nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT);

  nb::enum_<nixl_mem_t>(m, "MemoryType")
      .value("DRAM", nixl_mem_t::DRAM_SEG)
      .value("VRAM", nixl_mem_t::VRAM_SEG)
      .value("BLK", nixl_mem_t::BLK_SEG)
      .value("OBJ", nixl_mem_t::OBJ_SEG)
      .value("FILE", nixl_mem_t::FILE_SEG);

  nb::enum_<nixl_xfer_op_t>(m, "TransferOpType")
      .value("READ", nixl_xfer_op_t::NIXL_READ)
      .value("WRITE", nixl_xfer_op_t::NIXL_WRITE);

  nb::enum_<nixl_status_t>(m, "Status", R"nbdoc(
        Status code for NIXL operations.

        !! NOTE !! This differs from the C++ API. When an operation would
        return an error in the C++ API, here it would instead throw a similarly
        named exception. If a function can only one of these two statuses (usually
        SUCCESS) aside from throwing an error, the return code is omitted and is
        assumed to be that status code (again, usually SUCCESS).
      )nbdoc")
      .value("IN_PROG", nixl_status_t::NIXL_IN_PROG)
      .value("SUCCESS", nixl_status_t::NIXL_SUCCESS);

  nb::exception<nixlNotPostedError>(m, "NotPostedError");
  nb::exception<nixlInvalidParamError>(m, "InvalidParamError");
  nb::exception<nixlBackendError>(m, "BackendError");
  nb::exception<nixlNotFoundError>(m, "NotFoundError");
  nb::exception<nixlMismatchError>(m, "MismatchError");
  nb::exception<nixlNotAllowedError>(m, "NotAllowedError");
  nb::exception<nixlRepostActiveError>(m, "RepostActiveError");
  nb::exception<nixlUnknownError>(m, "UnknownError");
  nb::exception<nixlNotSupportedError>(m, "NotSupportedError");
  nb::exception<nixlRemoteDisconnectError>(m, "RemoteDisconnectError");
  nb::exception<nixlCanceledError>(m, "CanceledError");
  nb::exception<nixlNoTelemetryError>(m, "NoTelemetryError");

  nb::class_<nixlPyXferTelemetry>(m, "TransferTelemetry")
      .def_ro("startTime", &nixlPyXferTelemetry::startTime)
      .def_ro("postDuration", &nixlPyXferTelemetry::postDuration)
      .def_ro("xferDuration", &nixlPyXferTelemetry::xferDuration)
      .def_ro("totalBytes", &nixlPyXferTelemetry::totalBytes)
      .def_ro("descCount", &nixlPyXferTelemetry::descCount);

  nb::class_<nixl_xfer_dlist_t> xferDList(m, "TransferDescriptorList");
  bindDlist(xferDList);
  xferDList.def_prop_ro("type", &nixl_xfer_dlist_t::getType)
      .def_prop_ro("descriptor_count", &nixl_xfer_dlist_t::descCount)
      .def("is_empty", &nixl_xfer_dlist_t::isEmpty)
      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def(
          "__getitem__",
          [](nixl_xfer_dlist_t &self,
             size_t i) -> std::tuple<uintptr_t, size_t, uint64_t> {
            nixlBasicDesc desc = self[i];
            return {desc.addr, desc.len, desc.devId};
          },
          "idx"_a)
      .def(
          "__setitem__",
          [](nixl_xfer_dlist_t &self, size_t i,
             const std::tuple<uintptr_t, size_t, uint64_t> &desc) {
            auto [addr, len, devId] = desc;
            self[i] = nixlBasicDesc(addr, len, devId);
          },
          "idx"_a, "desc"_a)
      .def(
          "index",
          [](nixl_xfer_dlist_t &self, const nb::tuple &desc) {
            auto ret = self.getIndex(nixlBasicDesc(
                nb::cast<uintptr_t>(desc[0]), nb::cast<size_t>(desc[1]),
                nb::cast<uint64_t>(desc[2])));
            if (ret < 0)
              throw_nixl_exception((nixl_status_t)ret);
            return (int)ret;
          },
          "desc"_a)
      .def("remove", &nixl_xfer_dlist_t::remDesc, "idx"_a)
      .def("clear", &nixl_xfer_dlist_t::clear)
      .def("print", &nixl_xfer_dlist_t::print);

  nb::class_<nixl_reg_dlist_t> regDList(m, "RegistrationDescriptorList");
  bindDlist(regDList);
  regDList.def_prop_ro("type", &nixl_reg_dlist_t::getType)
      .def_prop_ro("descriptor_count", &nixl_reg_dlist_t::descCount)
      .def("is_empty", &nixl_reg_dlist_t::isEmpty)
      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def(
          "__getitem__",
          [](nixl_reg_dlist_t &self,
             size_t i) -> std::tuple<uintptr_t, size_t, uint64_t, std::string> {
            const nixlBlobDesc &desc = self[i];
            return {desc.addr, desc.len, desc.devId, desc.metaInfo};
          },
          "idx"_a)
      .def(
          "__setitem__",
          [](nixl_reg_dlist_t &self, unsigned int i,
             std::tuple<uintptr_t, size_t, uint64_t, std::string> desc) {
            auto [addr, len, devId, metaInfo] = std::move(desc);
            self[i] = nixlBlobDesc(addr, len, devId, std::move(metaInfo));
          },
          "idx"_a, "desc"_a)
      .def(
          "index",
          [](nixl_reg_dlist_t &self, const nb::tuple &desc) {
            int ret = self.getIndex(nixlBlobDesc(
                nb::cast<uintptr_t>(desc[0]), nb::cast<size_t>(desc[1]),
                nb::cast<uint64_t>(desc[2]), nb::cast<std::string>(desc[3])));
            if (ret < 0)
              throw_nixl_exception((nixl_status_t)ret);
            return ret;
          },
          "desc"_a)
      .def("trim", &nixl_reg_dlist_t::trim)
      .def("remove", &nixl_reg_dlist_t::remDesc, "idx"_a)
      .def("clear", &nixl_reg_dlist_t::clear)
      .def("print", &nixl_reg_dlist_t::print);

  nb::class_<nixlAgentConfig>(m, "AgentConfig")
      .def(nb::init<bool, bool, uint16_t, nixl_thread_sync_t, unsigned int,
                    uint64_t, uint64_t>(),
           "use_prog_thread"_a, "use_listen_thread"_a = false,
           "listen_port"_a = default_comm_port,
           "sync_mode"_a = nixl_thread_sync_t::NIXL_THREAD_SYNC_DEFAULT,
           "num_workers"_a = 1, "pthr_delay_us"_a = 0,
           "lthr_delay_us"_a = 100000);

  nb::class_<nixlAgent>(m, "Agent")
      .def(nb::init<std::string, nixlAgentConfig>(), "name"_a, "config"_a)
      .def("get_available_plugins",
           [](nixlAgent &self) -> std::vector<nixl_backend_t> {
             std::vector<nixl_backend_t> backends;
             throw_nixl_exception(self.getAvailPlugins(backends));
             return backends;
           })
      .def(
          "get_plugin_params",
          [](nixlAgent &self, const nixl_backend_t type)
              -> std::pair<nixl_b_params_t, std::vector<std::string>> {
            nixl_b_params_t params;
            nixl_mem_list_t mems;
            std::vector<std::string> memsVec;
            throw_nixl_exception(self.getPluginParams(type, mems, params));
            for (const auto &elm : mems)
              memsVec.emplace_back(nixlEnumStrings::memTypeStr(elm));
            return std::make_pair(params, memsVec);
          },
          "type"_a)
      .def(
          "get_backend_params",
          [](nixlAgent &self, uintptr_t backend)
              -> std::pair<nixl_b_params_t, std::vector<std::string>> {
            nixl_b_params_t params;
            nixl_mem_list_t mems;
            std::vector<std::string> memsVec;
            throw_nixl_exception(
                self.getBackendParams((nixlBackendH *)backend, mems, params));
            for (const auto &elm : mems)
              memsVec.emplace_back(nixlEnumStrings::memTypeStr(elm));
            return std::make_pair(params, memsVec);
          },
          "backend"_a)
      .def(
          "create_backend",
          [](nixlAgent &self, const nixl_backend_t &type,
             const nixl_b_params_t &initParams) -> uintptr_t {
            nixlBackendH *backend = nullptr;
            throw_nixl_exception(self.createBackend(type, initParams, backend));
            return (uintptr_t)backend;
          },
          "type"_a, "init_params"_a)
      .def(
          "register_memory",
          [](nixlAgent &self, nixl_reg_dlist_t descs,
             std::vector<uintptr_t> backends) -> nixl_status_t {
            nixl_opt_args_t extraParams;
            nixl_status_t ret;
            for (uintptr_t backend : backends)
              extraParams.backends.push_back((nixlBackendH *)backend);

            ret = self.registerMem(descs, &extraParams);
            throw_nixl_exception(ret);
            return ret;
          },
          "descs"_a, "backends"_a = std::vector<uintptr_t>())
      .def(
          "deregister_memory",
          [](nixlAgent &self, nixl_reg_dlist_t descs,
             std::vector<uintptr_t> backends) -> nixl_status_t {
            nixl_opt_args_t extraParams;
            nixl_status_t ret;
            for (uintptr_t backend : backends)
              extraParams.backends.push_back((nixlBackendH *)backend);

            ret = self.deregisterMem(descs, &extraParams);
            throw_nixl_exception(ret);
            return ret;
          },
          "descs"_a, "backends"_a = std::vector<uintptr_t>())
      .def(
          "query_memory",
          [](nixlAgent &self, nixl_reg_dlist_t descs,
             uintptr_t backend) -> std::vector<nixl_query_resp_t> {
            std::vector<nixl_query_resp_t> resp;
            nixl_opt_args_t extraParams;
            extraParams.backends.push_back((nixlBackendH *)backend);
            nixl_status_t ret = self.queryMem(descs, resp, &extraParams);
            throw_nixl_exception(ret);
            return resp;
          },
          "descs"_a, "backend"_a, R"nbdoc(
          Queries a registered descriptor list against a backend.

          Args:
              descs: The registered descriptors to query.
              backend: The backend handle to query against.

          Returns:
              A list of backend-specific responses for the descriptors.
          )nbdoc")
      .def(
          "make_connection",
          [](nixlAgent &self, const std::string &remote_agent,
             std::vector<uintptr_t> backends) {
            nixl_opt_args_t extraParams;

            for (uintptr_t backend : backends)
              extraParams.backends.push_back((nixlBackendH *)backend);

            nixl_status_t ret = self.makeConnection(remote_agent, &extraParams);
            throw_nixl_exception(ret);
            return ret;
          },
          "remote_agent"_a, "backends"_a)
      .def(
          "prep_transfer_descriptor_list",
          [](nixlAgent &self, std::string &agent_name,
             const nixl_xfer_dlist_t &descs,
             std::vector<uintptr_t> backends) -> uintptr_t {
            nixlDlistH *handle = nullptr;
            nixl_opt_args_t extraParams;

            for (uintptr_t backend : backends)
              extraParams.backends.push_back((nixlBackendH *)backend);

            throw_nixl_exception(
                self.prepXferDlist(agent_name, descs, handle, &extraParams));

            return (uintptr_t)handle;
          },
          "agent_name"_a, "descs"_a, "backend"_a = std::vector<uintptr_t>())
      .def(
          "make_transfer_request",
          [](nixlAgent &self, const nixl_xfer_op_t &operation,
             uintptr_t local_side, const std::vector<int> &local_indices,
             uintptr_t remote_side, const std::vector<int> &remote_indices,
             const std::string &notif_msg, std::vector<uintptr_t> backends,
             bool skip_desc_merge) -> uintptr_t {
            nixlXferReqH *handle = nullptr;
            nixl_opt_args_t extraParams;

            for (uintptr_t backend : backends)
              extraParams.backends.push_back((nixlBackendH *)backend);

            if (notif_msg.size() > 0) {
              extraParams.notifMsg = notif_msg;
              extraParams.hasNotif = true;
            }
            extraParams.skipDescMerge = skip_desc_merge;
            throw_nixl_exception(
                self.makeXferReq(operation, (nixlDlistH *)local_side,
                                 local_indices, (nixlDlistH *)remote_side,
                                 remote_indices, handle, &extraParams));

            return (uintptr_t)handle;
          },
          "operation"_a, "local_side"_a, "local_indices"_a, "remote_side"_a,
          "remote_indices"_a, "notif_msg"_a = std::string(),
          "backend"_a = std::vector<uintptr_t>(), "skip_desc_merg"_a = false)
      .def(
          "create_transfer_request",
          [](nixlAgent &self, const nixl_xfer_op_t &operation,
             const nixl_xfer_dlist_t &local_descs,
             const nixl_xfer_dlist_t &remote_descs,
             const std::string &remote_agent, const std::string &notif_msg,
             std::vector<uintptr_t> backends) -> uintptr_t {
            nixlXferReqH *handle = nullptr;
            nixl_opt_args_t extraParams;

            for (uintptr_t backend : backends)
              extraParams.backends.push_back((nixlBackendH *)backend);

            if (notif_msg.size() > 0) {
              extraParams.notifMsg = notif_msg;
              extraParams.hasNotif = true;
            }
            nixl_status_t ret =
                self.createXferReq(operation, local_descs, remote_descs,
                                   remote_agent, handle, &extraParams);

            throw_nixl_exception(ret);
            return (uintptr_t)handle;
          },
          "operation"_a, "local_descs"_a, "remote_descs"_a, "remote_agent"_a,
          "notif_msg"_a = std::string(), "backend"_a = std::vector<uintptr_t>())
      .def(
          "post_transfer_request",
          [](nixlAgent &self, uintptr_t request_handle,
             std::string notif_msg) -> nixl_status_t {
            nixl_opt_args_t extraParams;
            nixl_status_t ret;
            if (notif_msg.size() > 0) {
              extraParams.notifMsg = notif_msg;
              extraParams.hasNotif = true;
              ret = self.postXferReq((nixlXferReqH *)request_handle,
                                     &extraParams);
            } else {
              ret = self.postXferReq((nixlXferReqH *)request_handle);
            }
            throw_nixl_exception(ret);
            return ret;
          },
          "request_handle"_a, "notif_msg"_a = std::string())
      .def(
          "get_transfer_status",
          [](nixlAgent &self, uintptr_t reqh) -> nixl_status_t {
            nixl_status_t ret = self.getXferStatus((nixlXferReqH *)reqh);
            throw_nixl_exception(ret);
            return ret;
          },
          "request_handle"_a)
      .def(
          "get_transfer_telemetry",
          [](nixlAgent &self, uintptr_t reqh) -> nixlPyXferTelemetry {
            nixl_xfer_telem_t telemetry;
            throw_nixl_exception(
                self.getXferTelemetry((nixlXferReqH *)reqh, telemetry));
            return pyXferTelemetryFromCpp(telemetry);
          },
          "request_handle"_a)
      .def(
          "query_transfer_backend",
          [](nixlAgent &self, uintptr_t reqh) -> uintptr_t {
            nixlBackendH *backend = nullptr;
            throw_nixl_exception(
                self.queryXferBackend((nixlXferReqH *)reqh, backend));
            return (uintptr_t)backend;
          },
          "request_handle"_a)
      .def(
          "release_transfer_request",
          [](nixlAgent &self, uintptr_t reqh) -> nixl_status_t {
            nixl_status_t ret = self.releaseXferReq((nixlXferReqH *)reqh);
            throw_nixl_exception(ret);
            return ret;
          },
          "request_handle"_a)
      .def(
          "release_descriptor_list_handle",
          [](nixlAgent &self, uintptr_t handle) -> nixl_status_t {
            nixl_status_t ret = self.releasedDlistH((nixlDlistH *)handle);
            throw_nixl_exception(ret);
            return ret;
          },
          "handle"_a)
      .def(
          "get_notifs",
          [](nixlAgent &self, const std::vector<uintptr_t> &backends) {
            nixl_notifs_t newNotifs;
            nixl_opt_args_t extraParams;

            for (uintptr_t backend : backends)
              extraParams.backends.push_back((nixlBackendH *)backend);

            nixl_status_t ret = self.getNotifs(newNotifs, &extraParams);

            throw_nixl_exception(ret);

            std::map<std::string, nb::typed<nb::list, nb::bytes>> notifMap;
            for (const auto &[name, notifs] : newNotifs) {
              nb::list notifsPy;
              for (const auto &str : notifs)
                notifsPy.append(nb::bytes(str.data(), str.size()));
              notifMap[name] = notifsPy;
            }
            return notifMap;
          },
          "backends"_a = std::vector<uintptr_t>(), R"nbdoc(
        Returns a map from agent name to a list of notifications received from that agent.

        Optionally, a list of backends can be mentioned to only get those backends' notifications.
      )nbdoc")
      .def(
          "gen_notif",
          [](nixlAgent &self, const std::string &remote_agent,
             const std::string_view msg, std::vector<uintptr_t> backends) {
            nixl_opt_args_t extraParams;
            nixl_status_t ret;

            for (uintptr_t backend : backends)
              extraParams.backends.push_back((nixlBackendH *)backend);

            ret = self.genNotif(remote_agent, nixl_blob_t{msg}, &extraParams);

            throw_nixl_exception(ret);
            return ret;
          },
          "remote_agent"_a, "msg"_a, "backends"_a = std::vector<uintptr_t>())
      .def("get_local_metadata",
           [](nixlAgent &self) -> nb::bytes {
             // python can only interpret text strings
             std::string retStr;
             throw_nixl_exception(self.getLocalMD(retStr));
             return nb::bytes(retStr.data(), retStr.size());
           })
      .def(
          "get_local_partial_metadata",
          [](nixlAgent &self, nixl_reg_dlist_t descs, bool inc_conn_info,
             std::vector<uintptr_t> backends) -> nb::bytes {
            std::string retStr;

            nixl_opt_args_t extraParams;

            for (uintptr_t backend : backends)
              extraParams.backends.push_back((nixlBackendH *)backend);
            extraParams.includeConnInfo = inc_conn_info;

            throw_nixl_exception(
                self.getLocalPartialMD(descs, retStr, &extraParams));
            return nb::bytes(retStr.data(), retStr.size());
          },
          "descs"_a, "inc_conn_info"_a = false,
          "backends"_a = std::vector<uintptr_t>())
      .def(
          "load_remote_metadata",
          [](nixlAgent &self, const nb::bytes &agent_metadata) -> nb::bytes {
            // python can only interpret text strings
            std::string remoteName;
            throw_nixl_exception(self.loadRemoteMD(
                nixl_blob_t{agent_metadata.c_str(), agent_metadata.size()},
                remoteName));
            return nb::bytes(remoteName.data(), remoteName.size());
          },
          "agent_metadata"_a)
      .def("invalidate_remote_metadata", &nixlAgent::invalidateRemoteMD,
           "remote_name"_a)
      .def(
          "send_local_metadata",
          [](nixlAgent &self, std::string ip_addr, int port) {
            nixl_opt_args_t extraParams;

            extraParams.ipAddr = std::move(ip_addr);
            extraParams.port = port;

            throw_nixl_exception(self.sendLocalMD(&extraParams));
          },
          "ip_addr"_a = std::string(), "port"_a = 0)
      .def(
          "send_local_partial_metadata",
          [](nixlAgent &self, nixl_reg_dlist_t descs, bool inc_conn_info,
             std::vector<uintptr_t> backends, std::string ip_addr, int port) {
            nixl_opt_args_t extraParams;

            for (uintptr_t backend : backends)
              extraParams.backends.push_back((nixlBackendH *)backend);
            extraParams.includeConnInfo = inc_conn_info;
            extraParams.ipAddr = std::move(ip_addr);
            extraParams.port = port;

            throw_nixl_exception(self.sendLocalPartialMD(descs, &extraParams));
          },
          "descs"_a, "inc_conn_info"_a = false,
          "backends"_a = std::vector<uintptr_t>(), "ip_addr"_a = std::string(),
          "port"_a = 0)
      .def(
          "fetch_remote_metadata",
          [](nixlAgent &self, std::string remote_name, std::string ip_addr,
             int port) {
            nixl_opt_args_t extraParams;

            extraParams.ipAddr = std::move(ip_addr);
            extraParams.port = port;

            throw_nixl_exception(self.fetchRemoteMD(remote_name, &extraParams));
          },
          "remote_name"_a, "ip_addr"_a = std::string(), "port"_a = 0)
      .def(
          "invalidate_local_metadata",
          [](nixlAgent &self, std::string ip_addr, int port) {
            nixl_opt_args_t extraParams;

            extraParams.ipAddr = std::move(ip_addr);
            extraParams.port = port;

            throw_nixl_exception(self.invalidateLocalMD(&extraParams));
          },
          "ip_addr"_a = std::string(), "port"_a = 0)
      .def("check_remote_metadata", &nixlAgent::checkRemoteMD, "remote_name"_a,
           "descs"_a);
}

#else // !(__linux__ && __x86_64__)

// Non-linux-x86 stub: register an importable but empty nixl submodule.
// The NIXL transfer library is not available on this platform.
void M::Graph::Python::bindNIXLModule(nanobind::module_ m) {
  m.doc() =
      "NIXL bindings (unavailable: requires linux-x86_64 with the upstream "
      "NIXL shared library)";
}

#endif // __linux__ && __x86_64__
