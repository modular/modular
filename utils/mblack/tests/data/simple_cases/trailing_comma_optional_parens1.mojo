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
#   Path:   tests/data/simple_cases/trailing_comma_optional_parens1.py
#
# ===----------------------------------------------------------------------=== #

if e1234123412341234.winerror not in (
    _winapi.ERROR_SEM_TIMEOUT,
    _winapi.ERROR_PIPE_BUSY,
) or _check_timeout(t):
    pass

if x:
    if y:
        new_id = (
            max(
                Vegetable.objects.order_by("-id")[0].id,
                Mineral.objects.order_by("-id")[0].id,
            )
            + 1
        )


class X:
    def get_help_text(self):
        return ngettext(
            "Your password must contain at least %(min_length)d character.",
            "Your password must contain at least %(min_length)d characters.",
            self.min_length,
        ) % {"min_length": self.min_length}


class A:
    def b(self):
        if self.connection.mysql_is_mariadb and (
            10,
            4,
            3,
        ) < self.connection.mysql_version < (10, 5, 2):
            pass


# output

if e1234123412341234.winerror not in (
    _winapi.ERROR_SEM_TIMEOUT,
    _winapi.ERROR_PIPE_BUSY,
) or _check_timeout(t):
    pass

if x:
    if y:
        new_id = (
            max(
                Vegetable.objects.order_by("-id")[0].id,
                Mineral.objects.order_by("-id")[0].id,
            )
            + 1
        )


class X:
    def get_help_text(self):
        return ngettext(
            "Your password must contain at least %(min_length)d character.",
            "Your password must contain at least %(min_length)d characters.",
            self.min_length,
        ) % {"min_length": self.min_length}


class A:
    def b(self):
        if self.connection.mysql_is_mariadb and (
            10,
            4,
            3,
        ) < self.connection.mysql_version < (10, 5, 2):
            pass
