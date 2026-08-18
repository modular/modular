# Helpers to pre-populate a Bazel repository cache with artefacts fetched by
# Nix.  Bazel's cache layout is
#   <cache>/content_addressable/sha256/<sha256>/file
#   <cache>/content_addressable/sha256/<sha256>/id-<sha256(canonical id)>
# where the canonical id defaults to the space-joined list of URLs.
{ lib }:
{
  add =
    cache: seeds:
    lib.concatMapStrings (s: ''
      _d="${cache}/content_addressable/sha256/${s.sha256}"
      mkdir -p "$_d"
      cp --reflink=auto "${s.src}" "$_d/file"
      chmod 644 "$_d/file"
      : > "$_d/id-$(printf '%s' ${lib.escapeShellArg (lib.concatStringsSep " " s.urls)} | sha256sum | cut -d' ' -f1)"
    '') seeds;
  remove =
    cache: seeds:
    lib.concatMapStrings (s: ''
      rm -rf "${cache}/content_addressable/sha256/${s.sha256}"
    '') seeds;
}
