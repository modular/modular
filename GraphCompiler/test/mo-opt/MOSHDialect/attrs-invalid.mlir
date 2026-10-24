// RUN: mo-opt %s -verify-diagnostics

// expected-error @below{{expected all elements to have the same type}}
#foo = #mosh<ape[:scalar<si64> 1, :scalar<si8> ?, :scalar<si32> N]> : !mosh.ape
