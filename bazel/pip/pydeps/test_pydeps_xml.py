# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Tests for the JUnit ``test.xml`` output of ``pydeps_test``."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from pydeps_test import clean_text, write_junit_xml


def test_clean_text_strips_ansi() -> None:
    assert clean_text("\x1b[31mred\x1b[0m plain") == "red plain"
    assert clean_text("no codes here") == "no codes here"


def test_pass_case_emits_single_passing_testcase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml_path = tmp_path / "test.xml"
    monkeypatch.setenv("XML_OUTPUT_FILE", str(xml_path))

    write_junit_xml("//foo:bar", None, 0.5)

    root = ET.parse(xml_path).getroot()
    assert root.tag == "testsuite"
    assert root.get("name") == "//foo:bar"
    assert root.get("tests") == "1"
    assert root.get("failures") == "0"
    assert root.get("errors") == "0"
    assert root.get("time") == "0.500"
    # Exactly one testcase, no failure child.
    testcases = root.findall("testcase")
    assert len(testcases) == 1
    assert root.find(".//failure") is None


def test_failure_case_embeds_stripped_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml_path = tmp_path / "test.xml"
    monkeypatch.setenv("XML_OUTPUT_FILE", str(xml_path))

    write_junit_xml(
        "//foo:bar",
        "\x1b[31m//foo:bar has unused dependencies.\x1b[0m\n  //baz:qux",
        1.25,
    )

    root = ET.parse(xml_path).getroot()
    assert root.get("tests") == "1"
    assert root.get("failures") == "1"
    assert root.get("time") == "1.250"

    failure = root.find(".//failure")
    assert failure is not None
    assert failure.get("message") == "pydeps check failed"
    failure_text = failure.text
    assert failure_text is not None
    # ANSI color codes are stripped for XML compliance.
    assert "\x1b" not in failure_text
    assert "//foo:bar has unused dependencies." in failure_text
    assert "//baz:qux" in failure_text


def test_noop_when_xml_output_file_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XML_OUTPUT_FILE", raising=False)
    # Should not raise even though there is nowhere to write.
    write_junit_xml("//foo:bar", "some failure", 0.0)
