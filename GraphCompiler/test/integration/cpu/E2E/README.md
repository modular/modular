# Adding tests

To add a test to this folder go to:

`GraphCompiler/utils/test-helper/test_cases.py`

and implement your test in that. Once you've implemented it run:

`python3 GraphCompiler/utils/test-helper/export_mo_graph.py -n YOUR_CLASS_NAME`

This will create a MO graph and run line for your test. You can then add the
printed test to a new `.mlir` file in this folder.
