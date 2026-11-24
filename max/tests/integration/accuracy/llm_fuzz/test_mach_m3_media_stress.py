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
"""Checks the accepted-media workload against Mach's M3 request contract."""

import base64
import struct
import unittest

from scenarios.mach_m3_media_stress import (
    DATA,
    VIDEO_FIXTURES,
    MachM3ImageStress,
    MachM3VideoStress,
    mp4_part,
)


class MediaContractTest(unittest.TestCase):
    def test_item_cap(self) -> None:
        for scenario in (MachM3ImageStress(), MachM3VideoStress()):
            for index in range(8):
                with self.subTest(scenario=scenario.name, index=index):
                    self.assertTrue(
                        1 <= len(scenario.parts(index, "test")) <= 16
                    )

    def test_video_hints_and_cache_busting(self) -> None:
        for fixture in VIDEO_FIXTURES:
            with self.subTest(fixture=fixture):
                video = mp4_part(fixture, "cold")["video_url"]
                assert isinstance(video, dict)
                self.assertTrue(0.2 <= video["fps"] <= 5)
                self.assertEqual(video["max_long_side_pixel"] % 28, 0)
                prefix, encoded = video["url"].split(",", 1)
                self.assertEqual(prefix, "data:video/mp4;base64")
                original = (DATA / fixture).read_bytes()
                data = base64.b64decode(encoded)
                self.assertEqual(data[: len(original)], original)
                self.assertEqual(
                    data[len(original) :],
                    struct.pack(">I4s", 12, b"free") + b"cold",
                )
                other = mp4_part(fixture, "next")["video_url"]
                assert isinstance(other, dict)
                self.assertNotEqual(video["url"], other["url"])
                self.assertIn(b"avc1", original)
                self.assertIn(b"mdat", original)


if __name__ == "__main__":
    unittest.main()
