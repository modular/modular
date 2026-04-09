# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Mock package for testing stability marker warnings.
# This package is opted into stability markers (alongside "std"), so unstable
# APIs here will trigger warnings when used from outside the package.


@stable
struct StableStruct:
    """A stable struct that should not trigger warnings."""

    def __init__(out self):
        pass


struct UnstableStruct:
    """An unstable struct that should trigger warnings when used."""

    comptime UNSTABLE_CONST: Int = 99

    def __init__(out self):
        """Unstable constructor."""
        pass

    def unstable_method(self) -> Int:
        """An unstable method that should trigger warnings when called."""
        return 42

    comptime AliasedType = AnotherUnstableStruct


struct AnotherUnstableStruct:
    """A second unstable struct, exposed via a comptime alias in UnstableStruct.
    """

    def __init__(out self):
        pass

    @staticmethod
    def static_method() -> Int:
        return 0


@stable
def stable_fn():
    """A stable function that should not trigger warnings."""
    pass


def unstable_fn():
    """An unstable function that should trigger warnings when called."""
    pass


@stable
trait StableTrait:
    """A stable trait that should not trigger warnings when implemented."""

    pass


trait UnstableTrait:
    """An unstable trait that should trigger warnings when implemented."""

    pass


trait UnstableTraitWithMembers:
    """An unstable trait with a default-impl method and an associated comptime.
    """

    comptime ASSOC_TYPE = Int

    def default_method(self) -> Int:
        return 42


@stable
struct StructWithMethods:
    """A stable struct with both stable and unstable methods."""

    var value: Int

    @stable
    def __init__(out self):
        self.value = 0

    @stable
    def stable_method(self) -> Int:
        return self.value

    def unstable_method(self) -> Int:
        return self.value


@stable
trait TraitWithStableMethod:
    """A stable trait with a stable method requirement."""

    @stable
    def stable_required_method(self) -> Int:
        ...


@stable
trait TraitWithUnstableMethod:
    """A stable trait with an unstable method requirement."""

    def unstable_required_method(self) -> Int:
        ...


# Extension of UnstableStruct with an additional unstable method, for testing
# that @stable(recursive=True) suppresses warnings on extension-defined members.
__extension UnstableStruct:
    def extension_method(self) -> Int:
        """An unstable extension method on UnstableStruct."""
        return 99


# Test same-package usage: stable functions calling unstable functions internally.
# This should NOT trigger warnings because both are in the same opted-in package.
@stable
def stable_fn_using_unstable() -> Int:
    """A stable function that internally calls unstable APIs.

    When a user calls this stable function, they should NOT see warnings about
    the internal usage of unstable APIs. The intra-package usage is allowed.
    """
    # These internal calls to unstable APIs should NOT warn (same package).
    unstable_fn()
    var s = UnstableStruct()
    return s.unstable_method()


# Alias tests for escape hatches feature.
# Stable alias re-exporting an unstable struct - users should not get warning.
@stable
comptime StableAliasToUnstable = UnstableStruct

# Unstable alias - users should get warning when using this.
comptime UnstableAlias = StableStruct


# Stable constant alias.
@stable
comptime STABLE_CONSTANT: Int = 42

# Unstable constant alias.
comptime UNSTABLE_CONSTANT: Int = 100


# For testing: stable struct implementing stable trait with stable method.
# The struct's implementing method SHOULD be stable. If not, API author gets warning.
@stable
struct StableStructWithStableImpl(TraitWithStableMethod):
    @stable
    def stable_required_method(self) -> Int:
        return 1


# For testing: stable struct implementing stable trait with UNSTABLE method.
# This SHOULD warn the API author (always, not just with --warn-on-unstable-apis).
# The warning is tested by warn_api_author.mojo using FileCheck.
@stable
struct StableStructWithUnstableImpl(TraitWithStableMethod):
    # Not marked @stable - this should warn!
    def stable_required_method(self) -> Int:
        return 2


# For testing stable function return type check.
# This stable function returns an unstable type - should warn API author.
# The warning is tested by warn_api_author.mojo using FileCheck.
@stable
def stable_fn_returning_unstable() -> UnstableStruct:
    return UnstableStruct()


# For comparison: stable function returning stable type - no warning.
@stable
def stable_fn_returning_stable() -> StableStruct:
    return StableStruct()


# For testing: stable trait inheriting from unstable trait - should warn API author.
# This is invalid because a stable trait cannot have unstable parents.
# The warning is tested by warn_api_author.mojo using FileCheck.
@stable
trait StableTraitWithUnstableParent(UnstableTrait):
    pass


# For comparison: stable trait inheriting from stable trait - no warning.
@stable
trait StableTraitWithStableParent(StableTrait):
    pass


# ===----------------------------------------------------------------------===
# Alias stability tests for trait conformance
# ===----------------------------------------------------------------------===


@stable
trait TraitWithStableAlias:
    """A stable trait with a stable comptime alias requirement."""

    @stable
    # expected-note-re @below {{trait alias 'Value{{.*}}' in 'TraitWithStableAlias' is marked @stable}}
    comptime Value: Int


# For testing: stable struct implementing stable trait with UNSTABLE alias.
# This SHOULD warn the API author (analogous to StableStructWithUnstableImpl).
@stable
struct StableStructWithUnstableAliasImpl(TraitWithStableAlias):
    # Note: The alias name includes a suffix like `Value`1` due to internal naming.
    # expected-warning-re @below {{stable struct 'StableStructWithUnstableAliasImpl' implements stable trait alias 'Value{{.*}}' with unstable implementation}}
    comptime Value: Int = 42

    def __init__(out self):
        pass


# For comparison: stable struct with stable alias implementation - no warning.
@stable
struct StableStructWithStableAliasImpl(TraitWithStableAlias):
    @stable
    comptime Value: Int = 100

    def __init__(out self):
        pass


# ===----------------------------------------------------------------------===
# Error cases: @stable member in unstable struct/trait
# ===----------------------------------------------------------------------===
# These test that @stable members cannot be declared in unstable types.
# Since this package is opted-in, types without @stable are unstable.


struct UnstableStructWithStableMember:
    """An unstable struct that incorrectly has a @stable member."""

    def __init__(out self):
        pass

    # expected-warning@+1 {{@stable member cannot be declared in an unstable struct}}
    @stable
    def stable_method_in_unstable(self):
        pass


trait UnstableTraitWithStableMember:
    """An unstable trait that incorrectly has a @stable member."""

    # expected-warning@+1 {{@stable member cannot be declared in an unstable trait}}
    @stable
    def stable_method_in_unstable_trait(self):
        ...
