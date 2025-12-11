from collections import List
from os import Process

def test_process_run():
    var command = "echo"
    _ = Process.run(command, List[String]("== TEST_ECHO"))

fn main() raises:
    test_process_run()
