:title: C API
:sidebar_label: API index
:description: The MAX C API reference.
:lang: c

C API
=====

You can use the following C APIs to run inference with MAX Engine.

API headers
-----------

Each of the following pages represents one of the C API header files:

.. toctree::

   common
   context
   model
   tensor
   types
   value


Async API usage
---------------

Our C API allows for compiling and executing models asynchronously.  In general,
effective use of asynchronous APIs may be difficult, but rewarding for
performance.  To help with this, we're going to explain some important concepts
and mental models to keep in mind with the API.

Our APIs are async-safe unless stated otherwise, typically with a ```Sync``` in the
function identifier name.  For example, we have :cpp:any:`M_executeModel()` and
:cpp:any:`M_executeModelSync()`.

Types
+++++

Our API describes the underlying async-holding types with a "value or error"
concept.  Conceptually, this means that the type is in one of three states:

- ``Constructed`` - the value is not yet there, but there is no error
- ``Available`` - the value is there and ready for use
- ``Error`` - the value is not there and there is an error

Synchronization points
++++++++++++++++++++++

When using async APIs, it is a good idea to be mindful of the synchronization
point APIs currently provided below.  This is useful for discerning between the
``Constructed`` and ``Available`` states mentioned above.  After calling the
synchronization point, the input will never be in a ``Constructed`` state: it will
always resolve to either being ``Available`` or ``Error``.

- :cpp:any:`M_waitForCompilation()`
- :cpp:any:`M_waitForModel()`
- :cpp:any:`M_waitForTensors()`

Errors
++++++

Errors surface immediately when using our synchronous APIs.  Otherwise, in the
case of async APIs, errors will not surface until the next synchronization
point.  You can query the error message by calling :cpp:any:`M_getError()`.
