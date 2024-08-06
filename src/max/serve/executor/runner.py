# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


"""
ML execution abstractions.
"""

from abc import ABC, abstractmethod

from typing import Generic, TypeVar

Context = TypeVar("Context")
Tensor = TypeVar("Tensor")


class Runner(ABC, Generic[Context, Tensor]):
    """Interface for ML pipeline runners."""

    @property
    def inputs(self) -> list[Tensor]:
        ...

    @property
    def outputs(self) -> list[Tensor]:
        ...

    @abstractmethod
    async def init(self, context: Context) -> None:
        """Initialization hook, prepare lifetime setup here."""
        ...

    @abstractmethod
    async def teardown(self, context: Context) -> None:
        """Tear down resources for a runner."""
        ...

    @abstractmethod
    async def pre_run(self, context: Context) -> None:
        """Setup work prior to 'running'."""
        ...

    @abstractmethod
    async def run(self, context: Context) -> None:
        """Main method for 'running' a full runner."""
        ...

    @abstractmethod
    async def post_run(self, context: Context) -> None:
        """Cleanup work after 'running'."""
        ...
