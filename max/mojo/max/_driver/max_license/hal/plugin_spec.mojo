# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""The `MODULAR_DRIVER_PLUGINS` format.

The variable is a `;`-separated list of entries such as
`"CUDA@/a/CUDAPlugin.so;/b/CPUPlugin.so"`. An entry is `name@path`, or a bare
path that names no plugin. The name ends at the entry's first `@`. Whitespace
around a name or a path is dropped, and so are entries with no path.

Every Mojo HAL reader of the variable parses it here. The C++ readers parse the
same grammar in `MLRT/include/MLRT/Driver/DeviceContext/DriverPluginSpec.h`.
"""

from .status import HALError, STATUS_INVALID_ARG


@fieldwise_init
struct DriverPluginEntry(Copyable, Movable, Writable):
    """One plugin named in `MODULAR_DRIVER_PLUGINS`."""

    var name: String
    """The entry's name, or an empty string for a bare path."""

    var path: String
    """The path of the plugin's shared library."""

    @staticmethod
    def parse(spec: StringSpan) raises HALError -> Self:
        """Parses a spec that names exactly one plugin.

        Args:
            spec: A `MODULAR_DRIVER_PLUGINS` value.

        Returns:
            The one plugin entry in `spec`.

        Raises:
            `HALError` with `STATUS_INVALID_ARG` if `spec` names no plugin or
            more than one.
        """
        var entries = parse_driver_plugins(spec)
        if len(entries) != 1:
            raise HALError(
                STATUS_INVALID_ARG,
                message=String(
                    t"expected one driver plugin in '{spec}', found"
                    t" {len(entries)}"
                ),
            )
        return entries[0].copy()

    def write_to(self, mut writer: Some[Writer]):
        """Writes the entry in the form it parses from.

        Args:
            writer: The writer to write to.
        """
        if self.name:
            writer.write(self.name, "@")
        writer.write(self.path)


def parse_driver_plugins(spec: StringSpan) -> List[DriverPluginEntry]:
    """Parses a `MODULAR_DRIVER_PLUGINS` value.

    Args:
        spec: The value to parse.

    Returns:
        The plugin entries in `spec`, in order.
    """
    var entries = List[DriverPluginEntry]()
    for entry in spec.split(";"):
        var parts = entry.split("@", maxsplit=1)
        var name = String(parts[0].strip()) if len(parts) == 2 else String()
        var path = String(parts[len(parts) - 1].strip())
        if path:
            entries.append(DriverPluginEntry(name=name^, path=path^))
    return entries^
