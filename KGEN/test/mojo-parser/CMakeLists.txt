##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_custom_command(
  OUTPUT mojo_warmup_parser_cache.mlir mojo_warmup_parser_cache.input.mlir
  DEPENDS kgen-translate
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMAND echo "" > mojo_warmup_parser_cache.input.mlir
  COMMAND $<TARGET_FILE:kgen-translate> -import-mojo
                                        -o mojo_warmup_parser_cache.input.mlir
)
add_custom_target(warmup-mojo-parser-cache DEPENDS mojo_warmup_parser_cache.mlir)

add_modular_testsuite(check-mojo-parser
  TOP_LEVEL_CHECK check-mojo

  TOOLS
  kgen-opt
  hash-mlir
  warmup-mojo-parser-cache
  kgen-translate
)
