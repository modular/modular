##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Check for sem_timedwait. The semaphore implementation will use unnamed POSIX
# semaphores and sem_timedwait if they're available.
include(CheckSymbolExists)
if (UNIX AND NOT APPLE)
  # Linking with -pthread is required for semaphore symbol check to succeed.
  list(APPEND CMAKE_REQUIRED_LIBRARIES pthread)
endif ()
check_symbol_exists(sem_timedwait "semaphore.h" HAVE_SEM_TIMEDWAIT)
if (HAVE_SEM_TIMEDWAIT)
  add_definitions(-DMODULAR_HAVE_SEM_TIMEDWAIT)
endif ()

# On Apple systems, the semaphore implementation requires libdispatch, which
# is linked in through the system framework.
if (APPLE)
  list(APPEND LIBS System)
endif ()

add_modular_library(AsyncRTSupport
  STATIC OBJECT # Always generate an obj.${name} target.

  FileLocationDecoder.cpp
  Location.cpp
  Resource.cpp
  Semaphore.cpp
  ThreadAffinity.cpp
  TimerHeap.cpp
  UnknownLocationDecoder.cpp
  )
target_link_libraries(AsyncRTSupport PUBLIC ${LIBS} MThreading)
