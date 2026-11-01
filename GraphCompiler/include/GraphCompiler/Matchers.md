# Graph Compiler Pattern Matchers The Pattern Matcher utilities contained in

Matchers.h are a fork of the upstream mlir IR/Matchers.h to do create
template-metaprogrammed pattern options for fast pattern matching of MO graphs.
Examples of their use can be seen in PatternFusion.cpp and many other files in
the project.

## Logging Matches

For debuggability, the patterns have been heavily modified to accept a Logger
object which can print a backtrace after a branch of the pattern hits a match
failure.

An example of adding such a logger to the layer_norm pattern can be seen below
(with the layer_norm pattern locally modified to match a sum instead of a mean)

```cpp
    GraphCompiler::Logger log_tf;
    GraphCompiler::Logger log_pt;
    if (!t_layernorm.match(op.getResult(), &log_tf) &&
        !p_layernorm.match(op.getResult(), &log_pt)) {
      llvm::errs() << "\nMatch Failure Statck for TF"
                   << "\n\n";
      llvm::errs() << log_tf;
      llvm::errs() << "\nMatch Failure Statck for PT"
                   << "\n\n";
      llvm::errs() << log_pt;
```

If the Pattern fails to match, this print out things like this:

```mlir
Hit a Dead End at OpPattern mo.reduce.add! Rewinding Stack.
 %4 = mo.mean(%arg0, %0) : (!mo.tensor<[4, 3, 2], f32>,
       !mo.tensor<[], si64>) -> !mo.tensor<[4, 3, 1], f32>
  %5 = mo.static.broadcast_to(%4) : !mo.tensor<[4, 3, 1], f32>
        -> !mo.tensor<[4, 3, 2], f32>
   <block argument> of type '!mo.tensor<[4, 3, 2], f32>' at index: 0
    %6 = mo.sub(%arg0, %5) : !mo.tensor<[4, 3, 2], f32>
     %7 = mo.mul(%6, %6) : !mo.tensor<[4, 3, 2], f32>
      %8 = mo.mean(%7, %0) : (!mo.tensor<[4, 3, 2], f32>,
             !mo.tensor<[], si64>) -> !mo.tensor<[4, 3, 1], f32>
       %10 = mo.add(%8, %9) : !mo.tensor<[4, 3, 1], f32>
        %10 = mo.add(%8, %9) : !mo.tensor<[4, 3, 1], f32>
         %11 = mo.rsqrt(%10) : !mo.tensor<[4, 3, 1], f32>
          %12 = mo.static.broadcast_to(%11) : !mo.tensor<[4, 3, 1], f32>
                -> !mo.tensor<[4, 3, 2], f32>
           %14 = mo.mul(%12, %13) : !mo.tensor<[4, 3, 2], f32>
Hit a Dead End at OpPattern mo.rsqrt! Rewinding Stack.
 %2 = mo.constant <#M.dense_array<1.100000e+00, 2.100000e+00> : tensor<2xf32>>
        {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[2], f32>
  %13 = mo.static.broadcast_to(%2) : !mo.tensor<[2], f32>
          -> !mo.tensor<[4, 3, 2], f32>
   %12 = mo.static.broadcast_to(%11) : !mo.tensor<[4, 3, 1], f32>
         -> !mo.tensor<[4, 3, 2], f32>
    %14 = mo.mul(%12, %13) : !mo.tensor<[4, 3, 2], f32>
     %14 = mo.mul(%12, %13) : !mo.tensor<[4, 3, 2], f32>
      <block argument> of type '!mo.tensor<[4, 3, 2], f32>' at index: 0
       %15 = mo.mul(%arg0, %14) : !mo.tensor<[4, 3, 2], f32>
        %15 = mo.mul(%arg0, %14) : !mo.tensor<[4, 3, 2], f32>
         %19 = mo.add(%15, %18) : !mo.tensor<[4, 3, 2], f32>
          %19 = mo.add(%15, %18) : !mo.tensor<[4, 3, 2], f32>
```

Which shows a pattern failure for a `mo.reduce_add` pattern hitting a `mo.mean`
op. For more complex and branching patterns (i.e., a lot of `m_Optional` use)
not all branch failures will mean a pattern failure, but this should make
debugging complicated patterns significantly easier.
