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
from os.path import exists
from pathlib import Path

from testing import TestSuite, assert_false, assert_raises, assert_true


def create_and_delete(path: String):
    # verify that the test dir does not exist before starting the test
    assert_false(
        exists(path),
        "Unexpected dir " + path + " it should not exist",
    )

    os.mkdir(path, 0o777)
    assert_true(exists(path))

    os.rmdir(path)
    # trying to delete non existing dir
    with assert_raises(contains="Can not remove directory: "):
        os.rmdir(path)


def test_mkdir_and_rmdir(path: String):
    try:
        os.rmdir(path)
    except:
        pass
    # verify that the test dir does not exist before starting the test
    assert_false(
        exists(path),
        String("Unexpected dir ", path, " it should not exist"),
    )

    os.mkdir(path, 0o777)
    assert_true(exists(path))

    os.rmdir(path)
    # trying to delete non existing dir
    with assert_raises(contains="Can not remove directory: "):
        os.rmdir(path)


def test_mkdir_and_rmdir(path: Path):
    try:
        os.rmdir(path)
    except:
        pass
    # verify that the test dir does not exist before starting the test
    assert_false(
        exists(path), String("Unexpected dir ", path, " it should not exist")
    )

    os.mkdir(path, 0o777)
    assert_true(exists(path))

    os.rmdir(path)
    # trying to delete non existing dir
    with assert_raises(contains="Can not remove directory: "):
        os.rmdir(path)


def test_makedirs_and_removedirs(path: Path):
    try:
        os.removedirs(path)
    except:
        pass
    # verify that the test dir does not exist before starting the test
    assert_false(
        exists(path), String("Unexpected dir ", path, " it should not exist")
    )
    os.makedirs(path, exist_ok=True)
    assert_true(exists(path))
    with assert_raises():
        os.makedirs(path)
    # Make sure this doesn't throw error
    os.makedirs(path, exist_ok=True)
    os.removedirs(path)


def test_mkdir_mode():
    var my_dir_path = Path("my_dir")

    assert_false(
        exists(my_dir_path),
        "Unexpected dir " + my_dir_path.__fspath__() + " it should not exist",
    )

    # creating dir without writing permission
    os.mkdir(my_dir_path, 0o111)

    # TODO: This test is failing on Graviton internally in CI, revisit.
    # with assert_raises(contains="Permission denied"):
    #     var file = open(file_name, "w")
    #     file.close()
    #     if exists(file_name):
    #         remove(file_name)

    if exists(my_dir_path):
        os.rmdir(my_dir_path)


def test_rmdir_not_empty():
    var my_dir_path = Path("my_dir")
    var file_name = my_dir_path / "file.txt"

    assert_false(
        exists(my_dir_path),
        "Unexpected dir " + my_dir_path.__fspath__() + " it should not exist",
    )

    os.mkdir(my_dir_path)
    with open(file_name, "w"):
        pass

    with assert_raises(contains="Can not remove directory: "):
        os.rmdir(my_dir_path)

    os.remove(file_name)
    os.rmdir(my_dir_path)
    assert_false(exists(my_dir_path), "Failed to remove dir")


def test_all_mkdir_and_rmdir():
    test_mkdir_and_rmdir("my_dir")
    test_mkdir_and_rmdir(Path("my_dir"))
    if os.env.getenv("HOME") or os.env.getenv("USERPROFILE"):
        test_mkdir_and_rmdir(Path("~/my_dir").expanduser())


def test_all_makedirs_and_removedirs():
    test_makedirs_and_removedirs(os.path.join("dir1", "dir2", "dir3"))
    test_makedirs_and_removedirs(Path("dir1") / "dir2" / "dir3")


def main():
    var suite = TestSuite()

    suite.test[test_all_mkdir_and_rmdir]()
    suite.test[test_all_makedirs_and_removedirs]()
    suite.test[test_mkdir_mode]()
    suite.test[test_rmdir_not_empty]()

    suite^.run()
