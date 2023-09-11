##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
    TOP_LEVEL_CHECK
    check-support

    SUITE_NAME
    SupportHTTPUnitTests

    TEST_NAME
    SupportHTTPTests

    HTTPClientTest.cpp

    LINK_LIBS PRIVATE
    MSupportHTTP
)
