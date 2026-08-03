# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s

# A struct that conforms to a trait but mismatches an inherited defaulted
# method's signature must report a diagnostic, not crash.

# CHECK: error: 'Map[mapFn]' does not implement all requirements for 'Strategy'

@fieldwise_init
struct Map[Strat: Strategy, Dest: Copyable, //, mapFn: def (Strat.Value) thin -> Dest](Strategy):
    var strat: Self.Strat

    comptime Value = Self.Dest

    def value(mut self) raises -> Self.Value:
        return Self.mapFn(self.strat.value())


trait Strategy(Deinitable, Movable):
    comptime Value: Copyable

    def value(mut self) raises -> Self.Value:
        ...

    def map[To: Copyable, //, mapper: def (Self.Value) thin -> To](var self) -> Map[mapper]:
        return Map[mapper](self^)
