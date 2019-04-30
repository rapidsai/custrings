# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import pytest

import nvstrings

from utils import assert_eq


def test_cat():
    strs = nvstrings.to_device(["abc", "def", None, "", "jkl", "mno", "accént"])
    got = strs.cat()
    expected = ['abcdefjklmnoaccént']
    assert_eq(got, expected)

    # non-default separator
    got = strs.cat(sep=':')
    expected = ['abc:def::jkl:mno:accént']
    assert_eq(got, expected)

    # non default separator and na_rep
    got = strs.cat(sep=':', na_rep='_')
    expected = ['abc:def:_::jkl:mno:accént']
    assert_eq(got, expected)

    # non-null others, default separator, and na_rep
    got = strs.cat(["1", "2", "3", "4", "5", "é", None], sep=":", na_rep="_")
    expected = ['abc:1', 'def:2', '_:3', ':4', 'jkl:5', 'mno:é', 'accént:_']
    assert_eq(got, expected)

    # nvstrings others
    strs2 = nvstrings.to_device(["1", "2", "3", None, "5", "é", ""])
    got = strs.cat(strs2)
    expected = ['abc1', 'def2', None, None, 'jkl5', 'mnoé', 'accént']
    assert_eq(got, expected)


def test_join():
    strs = nvstrings.to_device(["1", "2", "3", None, "5", "é", ""])
    got = strs.join()
    expected = ['1235é']
    assert_eq(got, expected)

    # non-default sep
    got = strs.join(sep=':')
    expected = ['1:2:3:5:é:']
    assert_eq(got, expected)
