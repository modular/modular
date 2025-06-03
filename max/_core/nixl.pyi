# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum
from collections.abc import Mapping, Sequence
from typing import overload

from numpy.typing import ArrayLike

INIT_AGENT: str = ""

DEFAULT_COMM_PORT: int = 8888

class ThreadSyncMode(enum.Enum):
    NONE = 0

    STRICT = 1

    DEFAULT = 0

class MemoryType(enum.Enum):
    DRAM = 0

    VRAM = 1

    BLK = 2

    OBJ = 2

    FILE = 4

class TransferOpType(enum.Enum):
    READ = 0

    WRITE = 1

class Status(enum.Enum):
    IN_PROG = 1

    SUCCESS = 0

    ERR_NOT_POSTED = -1

    ERR_INVALID_PARAM = -2

    ERR_BACKEND = -3

    ERR_NOT_FOUND = -4

    ERR_MISMATCH = -5

    ERR_NOT_ALLOWED = -6

    ERR_REPOST_ACTIVE = -7

    ERR_UNKNOWN = -8

    ERR_NOT_SUPPORTED = -9

class NotPostedError(Exception):
    pass

class InvalidParamError(Exception):
    pass

class BackendError(Exception):
    pass

class NotFoundError(Exception):
    pass

class MismatchError(Exception):
    pass

class NotAllowedError(Exception):
    pass

class RepostActiveError(Exception):
    pass

class UnknownError(Exception):
    pass

class NotSupportedError(Exception):
    pass

class TransferDescriptorList:
    @overload
    def __init__(
        self, type: MemoryType, sorted: bool = False, init_size: int = 0
    ) -> None:
        """
        Constructs an empty descriptor list.

        Args:
          type: The type of memory each element describes
          sorted: Initial value of the 'sorted' field
          init_size: Initial capacity of the list
        """

    @overload
    def __init__(
        self,
        type: MemoryType,
        descs: Sequence[ArrayLike | tuple[int, int, int]],
        sorted: bool = False,
    ) -> None:
        """
        Constructs a descriptor list with given values.

        Args:
          type: The type of memory each element describes
          descs: A list of descriptors, each describing a section of memory.
                 Each element is either a tuple or a dlpack object.
          sorted: Whether to verify if the list is sorted or not.
        """

    def append(self, desc: ArrayLike | tuple[int, int, int]) -> None:
        """
        Adds a descriptor to the list.

        Args:
          desc: A descriptor describing a section of memory.
                Either a tuple or a dlpack object.
        """

    @property
    def type(self) -> MemoryType: ...
    @property
    def descriptor_count(self) -> int: ...
    def is_empty(self) -> bool: ...
    def is_sorted(self) -> bool: ...
    def __eq__(self, arg: object, /) -> bool: ...
    def __getitem__(self, idx: int) -> tuple[int, int, int]: ...
    def __setitem__(self, idx: int, desc: tuple[int, int, int]) -> None: ...
    def index(self, desc: tuple) -> int: ...
    def remove(self, idx: int) -> None: ...
    def verify_sorted(self) -> bool: ...
    def clear(self) -> None: ...
    def print(self) -> None: ...
    def __getstate__(self) -> bytes: ...
    def __setstate__(self, raw: bytes) -> None: ...

class RegistrationDescriptorList:
    @overload
    def __init__(
        self, type: MemoryType, sorted: bool = False, init_size: int = 0
    ) -> None:
        """
        Constructs an empty descriptor list.

        Args:
          type: The type of memory each element describes
          sorted: Initial value of the 'sorted' field
          init_size: Initial capacity of the list
        """

    @overload
    def __init__(
        self,
        type: MemoryType,
        descs: Sequence[ArrayLike | tuple[int, int, int, str]],
        sorted: bool = False,
    ) -> None:
        """
        Constructs a descriptor list with given values.

        Args:
          type: The type of memory each element describes
          descs: A list of descriptors, each describing a section of memory.
                 Each element is either a tuple or a dlpack object.
          sorted: Whether to verify if the list is sorted or not.
        """

    def append(self, desc: ArrayLike | tuple[int, int, int, str]) -> None:
        """
        Adds a descriptor to the list.

        Args:
          desc: A descriptor describing a section of memory.
                Either a tuple or a dlpack object.
        """

    @property
    def type(self) -> MemoryType: ...
    @property
    def descriptor_count(self) -> int: ...
    def is_empty(self) -> bool: ...
    def is_sorted(self) -> bool: ...
    def __eq__(self, arg: object, /) -> bool: ...
    def __getitem__(self, idx: int) -> tuple[int, int, int, str]: ...
    def __setitem__(
        self, idx: int, desc: tuple[int, int, int, str]
    ) -> None: ...
    def index(self, desc: tuple) -> int: ...
    def trim(self) -> TransferDescriptorList: ...
    def remove(self, idx: int) -> None: ...
    def verify_sorted(self) -> bool: ...
    def clear(self) -> None: ...
    def print(self) -> None: ...
    def __getstate__(self) -> bytes: ...
    def __setstate__(self, raw: bytes) -> None: ...

class AgentConfig:
    def __init__(
        self,
        use_prog_thread: bool,
        use_listen_thread: bool = False,
        listen_port: int = 0,
        sync_mode: ThreadSyncMode = ThreadSyncMode.NONE,
    ) -> None: ...

class Agent:
    def __init__(self, name: str, config: AgentConfig) -> None: ...
    def get_available_plugins(self) -> list[str]: ...
    def get_plugin_params(
        self, type: str
    ) -> tuple[dict[str, str], list[str]]: ...
    def get_backend_params(
        self, backend: int
    ) -> tuple[dict[str, str], list[str]]: ...
    def create_backend(
        self, type: str, init_params: Mapping[str, str]
    ) -> int: ...
    def register_memory(
        self, descs: RegistrationDescriptorList, backends: Sequence[int] = []
    ) -> Status: ...
    def deregister_memory(
        self, descs: RegistrationDescriptorList, backends: Sequence[int] = []
    ) -> Status: ...
    def make_connection(
        self, remote_agent: str, backends: Sequence[int]
    ) -> Status: ...
    def prep_transfer_descriptor_list(
        self,
        agent_name: str,
        descs: TransferDescriptorList,
        backend: Sequence[int] = [],
    ) -> int: ...
    def make_transfer_request(
        self,
        operation: TransferOpType,
        local_side: int,
        local_indices: Sequence[int],
        remote_side: int,
        remote_indices: Sequence[int],
        notif_msg: str = "",
        backend: Sequence[int] = [],
        skip_desc_merg: bool = False,
    ) -> int: ...
    def create_transfer_request(
        self,
        operation: TransferOpType,
        local_descs: TransferDescriptorList,
        remote_descs: TransferDescriptorList,
        remote_agent: str,
        notif_msg: str = "",
        backend: Sequence[int] = [],
    ) -> int: ...
    def post_transfer_request(
        self, request_handle: int, notif_msg: str = ""
    ) -> Status: ...
    def get_transfer_status(self, request_handle: int) -> Status: ...
    def query_transfer_backend(self, request_handle: int) -> int: ...
    def release_transfer_request(self, request_handle: int) -> Status: ...
    def release_descriptor_list_handle(self, handle: int) -> Status: ...
    def get_notifs(
        self, backends: Sequence[int] = []
    ) -> dict[str, list[bytes]]:
        """
        Returns a map from agent name to a list of notifications received from that agent.
        Optionally, a list of backends can be mentioned to only get those backends' notifications.
        """

    def gen_notif(
        self, remote_agent: str, msg: str, backends: Sequence[int] = []
    ) -> Status: ...
    def get_local_metadata(self) -> bytes: ...
    def get_local_partial_metadata(
        self,
        descs: RegistrationDescriptorList,
        inc_conn_info: bool = False,
        backends: Sequence[int] = [],
    ) -> bytes: ...
    def load_remote_metadata(self, agent_metadata: bytes) -> bytes: ...
    def invalidate_remote_metadata(self, remote_name: str) -> Status: ...
    def send_local_metadata(self, ip_addr: str = "", port: int = 0) -> None: ...
    def send_local_partial_metadata(
        self,
        descs: RegistrationDescriptorList,
        inc_conn_info: bool = False,
        backends: Sequence[int] = [],
        ip_addr: str = "",
        port: int = 0,
    ) -> None: ...
    def fetch_remote_metadata(
        self, remote_name: str, ip_addr: str = "", port: int = 0
    ) -> None: ...
    def invalidate_local_metadata(
        self, ip_addr: str = "", port: int = 0
    ) -> None: ...
    def check_remote_metadata(
        self, remote_name: str, descs: TransferDescriptorList
    ) -> Status: ...
