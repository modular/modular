##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-mojo

  SUITE_NAME
  KGENPackageUnitTests

  TEST_NAME
  KGENPackageTests

  PackageTest.cpp

  LINK_LIBS PRIVATE
  KGENPackage
)
