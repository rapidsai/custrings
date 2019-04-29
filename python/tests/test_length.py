# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import pytest

import nvstrings


def test_len():
    strs = nvstrings.to_device(
        ["abc", "Def", None, "jLl", "mnO", "PqR", "sTT", "dog and cat",
         "accénted", "", " 1234 ", "XYZ"])
    assert len(strs) == 12
    assert strs.len() == 12


def test_size():
    strs = nvstrings.to_device(
        ["abc", "Def", None, "jLl", "mnO", "PqR", "sTT", "dog and cat",
         "accénted", "", " 1234 ", "XYZ"])
    assert strs.size() == 12


def test_byte_count():
    strs = nvstrings.to_device(
        ["abc", "Def", None, "jLl", "mnO", "PqR", "sTT", "dog and cat",
         "accénted", "", " 1234 ", "XYZ"])
    assert strs.byt_count() == 47


def test_null_count():
    strs = nvstrings.to_device(
        ["abc", "Def", None, "jLl", "mnO", "PqR", "sTT", "dog and cat",
         "accénted", "", " 1234 ", "XYZ"])
    assert strs.null_count() == 1
