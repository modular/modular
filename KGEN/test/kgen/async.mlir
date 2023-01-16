// RUN: kgen %s -execute -func="call_it:index()" | FileCheck %s

// COM: All this code just adds 1 to 449500, giving 449501
// CHECK: --- 'call_it' returned 449501

kgen.func @slow_function(%arg0: index) -> !pop.coroutine<index> {
  %hdl = pop.coroutine.handle : <index>

  %lhs = index.constant 449500
  %result = index.add %lhs, %arg0

  %promise = pop.coroutine.promise %hdl : <index>
  %resPtr = pop.struct.gep %promise[0] : <struct<index>>
  pop.store %result, %resPtr : !pop.pointer<index>

  %idx1 = index.constant 1
  %ctxPtr = pop.offset %promise[%idx1] : !pop.pointer<struct<index>>
  %ctx = pop.pointer.bitcast %ctxPtr : !pop.pointer<struct<index>> to !pop.pointer<struct<pointer<scalar<invalid>>, i8>>
  %ctxOpaque = pop.pointer.bitcast %ctx : !pop.pointer<struct<pointer<scalar<invalid>>, i8>> to !pop.pointer<i8>
  pop.external_call @KGEN_CompilerRT_LLCL_Complete(%ctxOpaque) : (!pop.pointer<i8>) -> ()

  kgen.return %hdl : !pop.coroutine<index>
}

kgen.func @async_coroutine(%arg0: index) -> !pop.coroutine<index> {
  %idx1 = index.constant 1

  %hdl = pop.coroutine.handle : <index>
  %calleeHdl = kgen.call @slow_function(%arg0) : (index) -> !pop.coroutine<index>


  %promise = pop.coroutine.promise %hdl : <index>
  %ctxPtr = pop.offset %promise[%idx1] : !pop.pointer<struct<index>>
  %ctx = pop.pointer.bitcast %ctxPtr : !pop.pointer<struct<index>> to !pop.pointer<struct<pointer<scalar<invalid>>, i8>>
  %ctxOpaque = pop.pointer.bitcast %ctxPtr : !pop.pointer<struct<index>> to !pop.pointer<i8>

  %calleePromise = pop.coroutine.promise %calleeHdl : <index>
  %calleeCtxPtr = pop.offset %calleePromise[%idx1] : !pop.pointer<struct<index>>
  %calleeCtx = pop.pointer.bitcast %calleeCtxPtr : !pop.pointer<struct<index>> to !pop.pointer<struct<pointer<scalar<invalid>>, i8>>
  %calleeOpaqueCtx = pop.pointer.bitcast %calleeCtxPtr : !pop.pointer<struct<index>> to !pop.pointer<i8>

  %runtimePtr = pop.struct.gep %ctx[1] : <struct<pointer<scalar<invalid>>, i8>>
  %calleeRuntimePtr = pop.struct.gep %calleeCtx[1] : <struct<pointer<scalar<invalid>>, i8>>
  %runtime = pop.load %runtimePtr : !pop.pointer<i8>
  pop.store %runtime, %calleeRuntimePtr : !pop.pointer<i8>
  pop.coroutine.initialize %calleeHdl : <index>

  %resumeFn = kgen.addressof @__kgen_coro_resume : (!llvm.ptr<i8>) -> ()
  pop.coroutine.await {
    pop.external_call @KGEN_CompilerRT_LLCL_ExecuteAndResume(
      %resumeFn, %calleeHdl, %calleeOpaqueCtx, %hdl)
      : ((!llvm.ptr<i8>) -> (), !pop.coroutine<index>, !pop.pointer<i8>, !pop.coroutine<index>) -> ()
  }

  %calleeResPtr = pop.struct.gep %calleePromise[0] : <struct<index>>
  %result = pop.load %calleeResPtr : !pop.pointer<index>


  %resPtr = pop.struct.gep %promise[0] : <struct<index>>
  pop.store %result, %resPtr : !pop.pointer<index>
  pop.external_call @KGEN_CompilerRT_LLCL_Complete(%ctxOpaque) : (!pop.pointer<i8>) -> ()

  pop.coroutine.destroy %calleeHdl : <index>

  kgen.return %hdl : !pop.coroutine<index>
}

kgen.func @call_it() -> index {
  %arg0 = index.constant 1
  %nThreads = kgen.param.constant: i8 = <2>
  %runtime = pop.external_call @KGEN_CompilerRT_LLCL_CreateRuntime(%nThreads) : (i8) -> i8

  %hdl = kgen.call @async_coroutine(%arg0) : (index) -> !pop.coroutine<index>

  %promise = pop.coroutine.promise %hdl : <index>
  %one = index.constant 1
  %ctxPtrRaw = pop.offset %promise[%one] : !pop.pointer<struct<index>>
  %ctxPtr = pop.pointer.bitcast %ctxPtrRaw : !pop.pointer<struct<index>> to !pop.pointer<struct<pointer<scalar<invalid>>, i8>>
  %runtimePtr = pop.struct.gep %ctxPtr[1] : <struct<pointer<scalar<invalid>>, i8>>
  pop.store %runtime, %runtimePtr : !pop.pointer<i8>

  %ctxPtrVoid = pop.pointer.bitcast %ctxPtr : !pop.pointer<struct<pointer<scalar<invalid>>, i8>> to !pop.pointer<i8>
  pop.external_call @KGEN_CompilerRT_LLCL_InitializeContext(%ctxPtrVoid) : (!pop.pointer<i8>) -> ()

  %coroResume = kgen.addressof @__kgen_coro_resume : (!llvm.ptr<i8>) -> ()
  pop.external_call @KGEN_CompilerRT_LLCL_ExecuteAndWait(%coroResume, %hdl, %ctxPtr)
    : ((!llvm.ptr<i8>) -> (), !pop.coroutine<index>, !pop.pointer<struct<pointer<scalar<invalid>>, i8>>) -> ()

  %resultPtr = pop.struct.gep %promise[0] : <struct<index>>
  %result = pop.load %resultPtr : !pop.pointer<index>

  pop.coroutine.destroy %hdl : <index>

  pop.external_call @KGEN_CompilerRT_LLCL_DestroyRuntime(%runtime) : (i8) -> ()
  kgen.return %result : index
}

kgen.func @__kgen_coro_resume(%hdl: !llvm.ptr<i8>) {
  llvm.intr.coro.resume %hdl
  kgen.return
}

kgen.export [@call_it]
