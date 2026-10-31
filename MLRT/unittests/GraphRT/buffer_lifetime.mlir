// The unresolved argument keeps the executor's borrowed users list pending.
"test.function"() ({
^bb0(%input: si32):
  %result = "test.increment"(%input) : (si32) -> si32
  mef.output %result : si32
}) : () -> ()
