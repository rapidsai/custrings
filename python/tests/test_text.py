
import numpy as np
import nvstrings, nvtext

#
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True 
rmm.initialize()


def test_token_count():
    # default space delimiter
    strs = nvstrings.to_device(
        ["the quick brown fox jumped over the lazy brown dog",
         "the sable siamésé cat jumped under the brown sofa",
         None,
         ""]
    )
    outcome = nvtext.token_count(strs)
    expected = [10, 9, 0, 0]
    assert outcome == expected

    # custom delimiter
    outcome = nvtext.token_count(strs, delimiter='o')
    expected = [6, 3, 0, 0]
    assert outcome == expected

    # test device pointer
    outcome_darray = rmm.device_array(strs.size(), dtype=np.int32)
    nvtext.token_count(strs, devptr=d_arr.device_ctypes_pointer.value)
    expected = [10, 9, 0, 0]
    assert outcome_darray.copy_to_host() == expected


def test_unique_tokens():
    # default space delimiter
    strs = nvstrings.to_device(
        ["this is my favorite book",
         "Your Favorite book is different",
         None,
         ""]
    )
    unique_tokens_outcome = nvtext.unique_tokens(strs)
    expected = set(['Favorite', 'Your', 'book', 'different', 'favorite', 'is', 'my', 'this'])
    assert set(unique_tokens_outcome.to_host()) == expected

    # custom delimiter
    unique_tokens_outcome = nvtext.unique_tokens(strs, delimiter='my')
    expected = set([' favorite book', 'Your Favorite book is different', 'this is '])
    assert set(unique_tokens_outcome.to_host()) == expected


def test_contains_strings():
    strs = nvstrings.to_device(
        ["apples are green",
         "apples are a fruit",
         None,
         ""]
    )

    query_strings = nvstrings.to_device(['apple', 'fruit'])

    # host results
    contains_outcome = nvtext.contains_strings(strs, query_strings)
    expected = [
        [True, False],
        [True, True],
        [False, False],
        [False, False]
    ]
    assert contains_outcome == expected

    # device results
    outcome_darray = rmm.device_array((strs.size(), query_strings.size()),
                                      dtype=np.bool)
    nvtext.contains_strings(strs, query_strings,
                            devptr=outcome_darray.device_ctypes_pointer.value)
    assert np.array_equal(outcome_darray.copy_to_host(), expected)



print("strings_counts:",nvtext.strings_counts(strs,tokens))
d_arr = rmm.to_device(np.arange(strs.size()*tokens.size(),dtype=np.int32))
nvtext.strings_counts(strs,tokens,devptr=d_arr.device_ctypes_pointer.value)
print(" ",d_arr.copy_to_host())

#
print("contains_strings(cat,dog,bird,the):",nvtext.contains_strings(strs,['cat','dog','bird','the']))
print("strings_counts(cat,dog,bird,the):",nvtext.strings_counts(strs,['cat','dog','bird','the']))

#
strs = nvstrings.to_device(["kitten","kitton","kittén"])
print(strs)
print("edit_distance(kitten):",nvtext.edit_distance(strs,'kitten'))
print("edit_distance(kittén):",nvtext.edit_distance(strs,'kittén'))
strs1 = nvstrings.to_device(["kittén","sitting","Saturday","Sunday","book","back"])
print("s1:",strs1)
strs2 = nvstrings.to_device(["sitting","kitten","Sunday","Saturday","back","book"])
print("s2:",strs2)
print("edit_distance(s1,s2):",nvtext.edit_distance(strs1,strs2))
strs1 = None
strs2 = None

strs = None
tokens = None