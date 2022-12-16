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
#   Path:   tests/data/simple_cases/bracketmatch.py
#
# ===----------------------------------------------------------------------=== #

for ((x in {}) or {})["a"] in x:
    pass
pem_spam = lambda l, spam={"x": 3}: not spam.get(l.strip())
lambda x=lambda y={1: 3}: y["x" : lambda y: {1: 2}]: x


# output


for ((x in {}) or {})["a"] in x:
    pass
pem_spam = lambda l, spam={"x": 3}: not spam.get(l.strip())
lambda x=lambda y={1: 3}: y["x" : lambda y: {1: 2}]: x
