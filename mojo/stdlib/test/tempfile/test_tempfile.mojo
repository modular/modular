# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import os
from os.path import exists, split
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory, gettempdir, mkdtemp

from testing import assert_equal, assert_false, assert_true, TestSuite


def test_mkdtemp():
    var dir_name: String

    dir_name = mkdtemp()
    assert_true(exists(dir_name), "Failed to create temporary directory")
    os.rmdir(dir_name)
    assert_false(exists(dir_name), "Failed to delete temporary directory")

    dir_name = mkdtemp(prefix="my_prefix", suffix="my_suffix")
    assert_true(exists(dir_name), "Failed to create temporary directory")
    var name = dir_name.split(os.sep)[-1]
    assert_true(name.startswith("my_prefix"))
    assert_true(name.endswith("my_suffix"))

    os.rmdir(dir_name)
    assert_false(exists(dir_name), "Failed to delete temporary directory")

    dir_name = mkdtemp(dir=Path().__fspath__())
    assert_true(exists(dir_name), "Failed to create temporary directory")
    assert_true(
        exists(Path() / dir_name.split(os.sep)[-1]),
        "Expected directory to be created in cwd",
    )
    os.rmdir(dir_name)
    assert_false(exists(dir_name), "Failed to delete temporary directory")


struct TempEnvWithCleanup:
    var vars_to_set: Dict[String, String]
    var _vars_back: Dict[String, String]
    var clean_up_function: fn () raises -> None
    """Function called after the context manager exits if an error occurs."""

    fn __init__(
        out self,
        vars_to_set: Dict[String, String],
        clean_up_function: fn () raises -> None,
    ):
        self.vars_to_set = vars_to_set.copy()
        self._vars_back = Dict[String, String]()
        self.clean_up_function = clean_up_function

    def __enter__(mut self):
        for key_value in self.vars_to_set.items():
            var key = key_value.key
            var value = key_value.value
            self._vars_back[key] = os.getenv(key)
            _ = os.setenv(key, value, overwrite=True)

    fn __exit__(mut self):
        for key_value in self.vars_to_set.items():
            var key = key_value.key
            var value = key_value.value
            _ = os.setenv(key, value, overwrite=True)

    def __exit__(mut self, error: Error) -> Bool:
        self.__exit__()
        self.clean_up_function()
        return False


def _clean_up_gettempdir_test():
    var dir_without_writing_access = Path() / "dir_without_writing_access"
    if exists(dir_without_writing_access):
        os.rmdir(dir_without_writing_access)
    var dir_with_writing_access = Path() / "dir_with_writing_access"
    if exists(dir_with_writing_access):
        os.rmdir(dir_with_writing_access)


def _set_up_gettempdir_test(
    dir_with_writing_access: Path, dir_without_writing_access: Path
):
    os.mkdir(dir_with_writing_access, mode=0o700)
    try:
        os.mkdir(dir_without_writing_access, mode=0o100)
    except:
        os.rmdir(dir_with_writing_access)
        raise Error(
            "Failed to setup test, couldn't create ", dir_without_writing_access
        )


def test_gettempdir():
    var non_existing_dir = Path() / "non_existing_dir"
    assert_false(
        exists(non_existing_dir), String("Unexpected dir", non_existing_dir)
    )
    var dir_without_writing_access = Path() / "dir_without_writing_access"
    var dir_with_writing_access = Path() / "dir_with_writing_access"
    _set_up_gettempdir_test(dir_with_writing_access, dir_without_writing_access)

    var tmpdir_result: Optional[String]
    var vars_to_set = Dict[String, String]()

    # test TMPDIR is used first
    vars_to_set["TMPDIR"] = String(dir_with_writing_access)
    with TempEnvWithCleanup(
        vars_to_set,
        _clean_up_gettempdir_test,
    ):
        tmpdir_result = gettempdir()
        assert_true(tmpdir_result, "Failed to get temporary directory")
        assert_equal(
            tmpdir_result.value(),
            String(dir_with_writing_access),
            String("expected to get:", dir_with_writing_access),
        )

    # test gettempdir falls back to TEMP
    vars_to_set["TMPDIR"] = String(non_existing_dir)
    vars_to_set["TEMP"] = String(dir_with_writing_access)
    with TempEnvWithCleanup(
        vars_to_set,
        _clean_up_gettempdir_test,
    ):
        tmpdir_result = gettempdir()
        assert_true(tmpdir_result, "Failed to get temporary directory")
        assert_equal(
            tmpdir_result.value(),
            String(dir_with_writing_access),
            String("expected to get:", dir_with_writing_access),
        )

    # test gettempdir falls back to TMP
    vars_to_set["TMPDIR"] = String(non_existing_dir)
    vars_to_set["TEMP"] = String(non_existing_dir)
    vars_to_set["TMP"] = String(dir_with_writing_access)
    with TempEnvWithCleanup(
        vars_to_set,
        _clean_up_gettempdir_test,
    ):
        tmpdir_result = gettempdir()
        assert_true(tmpdir_result, "Failed to get temporary directory")
        assert_equal(
            tmpdir_result.value(),
            String(dir_with_writing_access),
            String("expected to get:", dir_with_writing_access),
        )

    _clean_up_gettempdir_test()


def test_temporary_directory() -> None:
    var tmp_dir2 = String()
    with TemporaryDirectory(suffix="my_suffix", prefix="my_prefix") as tmp_dir:
        assert_true(exists(tmp_dir), "Failed to create temp dir " + tmp_dir)
        assert_true(tmp_dir.endswith("my_suffix"))
        assert_true(tmp_dir.split(os.sep)[-1].startswith("my_prefix"))
        tmp_dir2 = tmp_dir
    assert_false(exists(tmp_dir2), "Failed to delete temp dir " + tmp_dir2)

    with TemporaryDirectory() as tmp_dir:
        assert_true(exists(tmp_dir), "Failed to create temp dir " + tmp_dir)
        _ = open(Path(tmp_dir) / "test_file", "w")
        os.mkdir(Path(tmp_dir) / "test_dir")
        _ = open(Path(tmp_dir) / "test_dir" / "test_file2", "w")
        tmp_dir2 = tmp_dir
    assert_false(exists(tmp_dir2), "Failed to delete temp dir " + tmp_dir2)


def test_named_temporary_file_deletion():
    var tmp_file: NamedTemporaryFile
    var file_path: String

    with NamedTemporaryFile(
        prefix="my_prefix", suffix="my_suffix", dir=Path().__fspath__()
    ) as my_tmp_file:
        file_path = my_tmp_file.name
        var file_name = file_path.split(os.sep)[-1]
        assert_true(exists(file_path), "Failed to create file " + file_path)
        assert_true(file_name.startswith("my_prefix"))
        assert_true(file_name.endswith("my_suffix"))
        assert_equal(split(file_path)[0], Path().__fspath__())
    assert_false(exists(file_path), "Failed to delete file " + file_path)

    with NamedTemporaryFile(delete=False) as my_tmp_file:
        file_path = my_tmp_file.name
        assert_true(exists(file_path), "Failed to create file " + file_path)
    assert_true(exists(file_path), "File " + file_path + " should still exist")
    os.remove(file_path)

    tmp_file = NamedTemporaryFile()
    file_path = tmp_file.name
    assert_true(exists(file_path), "Failed to create file " + file_path)
    tmp_file.close()
    assert_false(exists(file_path), "Failed to delete file " + file_path)

    tmp_file = NamedTemporaryFile(delete=False)
    file_path = tmp_file.name
    assert_true(exists(file_path), "Failed to create file " + file_path)
    tmp_file.close()
    assert_true(exists(file_path), "File " + file_path + " should still exist")
    os.remove(file_path)


def test_named_temporary_file_write():
    var file_name: String
    var contents: String

    with NamedTemporaryFile(delete=False) as my_tmp_file:
        file_name = my_tmp_file.name
        my_tmp_file.write("hello world")

    with open(file_name, "r") as my_file:
        contents = my_file.read()
    assert_equal("hello world", contents)
    os.remove(file_name)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
