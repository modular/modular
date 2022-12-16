# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   tests/data/miscellaneous/async_as_identifier.py
#
# ===----------------------------------------------------------------------=== #


def async():
    pass


def await():
    pass


await = lambda: None
async = lambda: None
async()
await()


def sync_fn():
    await = lambda: None
    async = lambda: None
    async()
    await()


async def async_fn():
    await async_fn()


# output
def async():
    pass


def await():
    pass


await = lambda: None
async = lambda: None
async()
await()


def sync_fn():
    await = lambda: None
    async = lambda: None
    async()
    await()


async def async_fn():
    await async_fn()
