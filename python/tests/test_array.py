# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import nvstrings


def test_gather():
    strs = nvstrings.to_device(["abc", "defghi", None, "cat"])
    got = strs.gather([1, 3, 2])
    expected = ['defghi', 'cat', None]
    assert got.to_host() == expected


def test_gather_bool():
    strs = nvstrings.to_device(["abc", "defghi", None, "cat"])
    got = strs.gather([True, False, False, True])
    expected = ['abc', 'cat']
    assert got.to_host() == expected


def test_sublist():
    strs = nvstrings.to_device(["abc", "defghi", None, "cat"])
    got = strs.sublist([1, 3, 2])
    expected = ['defghi', 'cat', None]
    assert got.to_host() == expected


def test_remove_strings():
    strs = nvstrings.to_device(["abc", "defghi", None, "cat"])
    got = strs.remove_strings([0, 2])
    expected = ['defghi', 'cat']
    assert got.to_host() == expected
