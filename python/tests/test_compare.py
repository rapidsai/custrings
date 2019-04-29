# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import pytest
import numpy as np

import nvstrings

from librmm_cffi import librmm as rmm

from utils import assert_eq


def test_compare():
    strs = nvstrings.to_device(
        ["hello", "there", "world", "accéntéd", None, ""])
    got = strs.compare('there')
    expected = [-12, 0, 3, -19, None, -1]
    assert_eq(got, expected)

    # device array
    arr = np.arange(strs.size(), dtype=np.int32)
    d_arr = rmm.to_device(arr)
    devmem = d_arr.device_ctypes_pointer.value
    strs.compare("there", devmem)
    expected = [-12, 0, 3, -19, -1, -1]
    assert_eq(d_arr.copy_to_host().tolist(), expected)


def test_find():
    strs = nvstrings.to_device(
        ["hello", "there", "world", "accéntéd", None, ""])
    got = strs.find("o")
    expected = [4, -1, 1, -1, None, -1]
    assert_eq(got, expected)


def test_find_from():
    strs = nvstrings.to_device(
        ["hello", "there", "world", "accéntéd", None, ""])
    got = strs.find_from("r")
    expected = [-1, 3, 2, -1, None, -1]
    assert_eq(got, expected)


def test_rfind():
    strs = nvstrings.to_device(
        ["hello", "there", "world", "accéntéd", None, ""])
    got = strs.rfind("d")
    expected = [-1, -1, 4, 7, None, -1]
    assert_eq(got, expected)


def test_find_multiple():
    strs = nvstrings.to_device(
        ["hello", "there", "world", "accéntéd", None, ""])
    got = strs.find_multiple(['e', 'o', 'd'])
    expected = [[1, 4, -1], [2, -1, -1], [-1, 1, 4], [-1, -1, 7],
                [None, None, None], [-1, -1, -1]]
    assert_eq(got, expected)


def test_startswith():
    strs = nvstrings.to_device(
        ["hello", "there", "world", "accéntéd", None, ""])
    got = strs.startswith("he")
    expected = [True, False, False, False, None, False]
    assert_eq(got, expected)


def test_endswith():
    strs = nvstrings.to_device(
        ["hello", "there", "world", "accéntéd", None, ""])
    strs.endswith("d")
    expected = [False, False, True, True, None, False]
    assert_eq(got, expected)


def test_match():
    strs = nvstrings.to_device(["tempo", "there", "this", "ether", None, ""])
    got = strs.match("th")
    expected = [False, True, True, False, None, False]
    assert_eq(got, expected)


def test_index():
    strs = nvstrings.to_device(
        ["he-llo", "-there-", "world-", "accént-éd", None, "-"])
    got = strs.index("-")
    expected = [2, 0, 5, 6, None, 0]
    assert_eq(got, expected)


def test_rindex():
    strs = nvstrings.to_device(
        ["he-llo", "-there-", "world-", "accént-éd", None, "-"])
    got = strs.rindex("-")
    expected = [2, 6, 5, 6, None, 0]
    assert_eq(got, expected)


def test_contains():
    strs = nvstrings.to_device(
        ["he-llo", "-there-", "world-", "accént-éd", None, "-"])
    got = strs.contains("l")
    expected = [True, False, True, False, None, False]
    assert_eq(got, expected)
