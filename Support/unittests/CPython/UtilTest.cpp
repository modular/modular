//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// TODO: Ideally we wouldn't need this
#undef Py_LIMITED_API

#include "Support/CPython/Util.h"
#include <Python.h>
#include <gtest/gtest.h>

using namespace M::CPython;

class UtilTest : public ::testing::Test {
protected:
  void SetUp() override { Py_Initialize(); }

  void TearDown() override { Py_Finalize(); }
};

TEST_F(UtilTest, SetDictKeyValueBool) {
  PyObject *dict = PyDict_New();
  ASSERT_TRUE(dict != nullptr);

  EXPECT_TRUE(setDictKeyValueBool(dict, "test_key", true));
  PyObject *value = PyDict_GetItemString(dict, "test_key");
  ASSERT_TRUE(value != nullptr);
  EXPECT_EQ(value, Py_True);

  Py_DECREF(dict);
}

TEST_F(UtilTest, SetDictKeyValueLong) {
  PyObject *dict = PyDict_New();
  ASSERT_TRUE(dict != nullptr);

  EXPECT_TRUE(setDictKeyValueLong(dict, "test_key", 42L));
  PyObject *value = PyDict_GetItemString(dict, "test_key");
  ASSERT_TRUE(value != nullptr);
  EXPECT_TRUE(PyLong_Check(value));
  EXPECT_EQ(PyLong_AsLong(value), 42L);

  Py_DECREF(dict);
}

TEST_F(UtilTest, SetDictKeyValueString) {
  PyObject *dict = PyDict_New();
  ASSERT_TRUE(dict != nullptr);

  EXPECT_TRUE(setDictKeyValueString(dict, "test_key", "test_value"));
  PyObject *value = PyDict_GetItemString(dict, "test_key");
  ASSERT_TRUE(value != nullptr);
  EXPECT_TRUE(PyUnicode_Check(value));
  EXPECT_STREQ(PyUnicode_AsUTF8(value), "test_value");

  Py_DECREF(dict);
}

TEST_F(UtilTest, GetDictValue) {
  PyObject *dict = PyDict_New();
  ASSERT_TRUE(dict != nullptr);

  PyObject *pyValue = PyLong_FromLong(42L);
  PyDict_SetItemString(dict, "test_key", pyValue);

  auto wrapper = getDictValue(dict, "test_key");
  ASSERT_TRUE(wrapper);
  EXPECT_EQ(PyLong_AsLong(wrapper.ptr), 42L);

  Py_DECREF(pyValue);
  Py_DECREF(dict);
}

TEST_F(UtilTest, GetDictBool) {
  PyObject *dict = PyDict_New();
  ASSERT_TRUE(dict != nullptr);

  PyDict_SetItemString(dict, "test_key", Py_True);

  auto result = getDictBool(dict, "test_key");
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result.value());

  Py_DECREF(dict);
}

TEST_F(UtilTest, GetDictValueAsBool) {
  PyObject *dict = PyDict_New();
  ASSERT_TRUE(dict != nullptr);

  // Test existing key
  PyDict_SetItemString(dict, "test_key", Py_True);
  auto result = getDictValueAs<bool>(dict, "test_key");
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result.value());

  // Test non-existent key
  auto nonExistentResult = getDictValueAs<bool>(dict, "non_existent_key");
  EXPECT_FALSE(nonExistentResult.has_value());

  Py_DECREF(dict);
}

TEST_F(UtilTest, AsString) {
  PyObject *pyStr = PyUnicode_FromString("test");
  auto result = asString(pyStr);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "test");

  Py_DECREF(pyStr);
}

TEST_F(UtilTest, StringToPythonObject) {
  auto wrapper = stringToPythonObject("test");
  ASSERT_TRUE(wrapper);
  EXPECT_TRUE(PyUnicode_Check(wrapper.ptr));
  EXPECT_STREQ(PyUnicode_AsUTF8(wrapper.ptr), "test");
}

TEST_F(UtilTest, InvalidDictOperations) {
  EXPECT_FALSE(setDictKeyValueBool(nullptr, "test_key", true));
  EXPECT_FALSE(setDictKeyValueLong(nullptr, "test_key", 42L));
  EXPECT_FALSE(setDictKeyValueString(nullptr, "test_key", "test_value"));
  EXPECT_FALSE(getDictValue(nullptr, "test_key"));
  EXPECT_FALSE(getDictBool(nullptr, "test_key").has_value());
  EXPECT_FALSE(getDictValueAs<bool>(nullptr, "test_key").has_value());
}
