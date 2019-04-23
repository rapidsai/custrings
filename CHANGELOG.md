# cuStrings/nvStrings 0.7.0 (TBD)

Version jump from 0.3->0.7 is to align with other RAPIDS projects.

## New Features

- PR #195 Added IPC python bindings
- PR #204 Added stol/ltos and stod/dtos number converters
- PR #194 Added method to create strings from array of floats
- PR #193 Added IPC transfer methods to C++ interface
- PR #188 Rename rave to nvtext and publish in nvstrings package
- PR #180 Added edit-distance to rave module
- PR #176 Added match_strings to nvstrings
- PR #172 Added to/from boolean conversion methods
- PR #171 Added conversion to/from subset of ISO8601 format
- PR #230 Add ngrams function to nvtext module

## Improvements

- PR #255 Removed unnecessary cudaDeviceSync calls
- PR #237 Removed internal util.h from include dir
- PR #216 Fixed build instructions and removed obsolete doc/build files
- PR #178 Added compute_statistics to C++ interface
- PR #173 Added flake8 checks to for /python files
- PR #167 Added doxygen to NVStrings.h and NVCategory.h
- PR #183 Align version to 0.7
- PR #192 Rework CMakeLists and separate RMM, update conda recipe and setup.py
- PR #181 Update python docstrings to numpydoc style
- PR #196 Include nvtext in python module setup
- PR #221 Create separate conda packages for libnvstrings and nvstrings

## Bug Fixes

- PR #248 Fixed docstring for index,rindex,find,rfind
- PR #245 Fixed backref to continue replacing
- PR #234 Added more type-checking to gather method
- PR #226 Added data pre-check to create_from_index
- PR #218 Parameter check corrected for pad methods
- PR #217 Corrected custring.cu method signatures
- PR #213 Fixing README link to Python API docs
- PR #211 Re-ordering 2 methods stops invalid-dev-fn when called from cudf
- PR #207 Fixed handling empty pattern in findall/count
- PR #202 Allow insert on start=stop in replace
- PR #201 Fixed some pad examples in nvstrings.py doc comments
- PR #186 Fixed memory error in booleans to strings method
- PR #235 Fix anaconda upload script for new conda recipe
- PR #239 Fix definitions of upload files for conda packages
- PR #241 Revert split package to individual recipes to avoid run_exports bug


# cuStrings/nvStrings 0.3.0 (15 Mar 2019)

## New Features

- PR #148 Added fillna method to nvstrings
- PR #105 Added ip2int/int2ip method for converting IPv4 addresses to and from ints
- PR #99 Added nvstrings.itos (int to string) method to nvstrings
- PR #94 Support backreferences in regexes
- PR #81 Added copy to nvstrings
- PR #79 Added add_strings to nvstrings bindings
- PR #67 Adding merge_and_remap
- PR #65 Implement len() for nvstrings
- PR #46 Added htoi (hex to int) to nvstrings
- PR #42 Strip multiple chars
- PR #41 Added from_offsets method
- PR #23 Add [] operator to nvstrings
- PR #9 category merge python interface
- PR #1 Added nvcategory.merge_category
- PR #154 Default to CXX11_ABI=ON

## Improvements

- PR #163 Follow standard RAPIDS README format
- PR #152 Add CHANGELOG.md
- PR #145 Improve handling of whitespace in split
- PR #144 Allow -1 for gather() position values
- PR #141 Allow for null category values
- PR #137 Update gpuCI scripts
- PR #134 gather/and_remap
- PR #130 Add bind_cpointer to nvcategory, add own parameter
- PR #127 Splitup 6k-line source file
- PR #125 Added nullfirst param to sort/order
- PR #124 Split to return column of nulls if all values null
- PR #121 Added nulls param to itos
- PR #110 Rename split, rsplit, extract methods to match expected behavior
- PR #108 Return None for nulls when returning Python lists
- PR #106 Added set_null_bitmask method
- PR #102 Added gather method
- PR #101 Support both single instance, and lists in from_strings
- PR #100 Support empty string '' for na_rep for nvstrings.cat
- PR #92 Use global memory for long regexes
- PR #89 Added remove-unused-keys
- PR #88 Added get_value_bounds
- PR #86 Add gather handle for ndarray
- PR #84 Added accessor for c-pointers
- PR #75 return null-bitmask in create_offsets
- PR #71 Added null_count method
- PR #68 Add byte_count to nvstrings interface
- PR #62 add regex quantifier documentation
- PR #61 add quantifier support for regexes
- PR #60 add support for ndarray, buffer in getitem/gather
- PR #49 nvcategory update keys methods: add, remove, set
- PR #54 throw exception if enumerate is used on nvstrings instance
- PR #39 Rename sublist1 to gather
- PR #12, #13, #14, #136 RMM integration
- PR #2 Refactored Reclass

## Bug Fixes

- PR #151 Update conda build script to handle build changes
- PR #150 split-ws var init corrected
- PR #140 Fix null placement in from_strings
- PR #133 Corrected parameter types
- PR #126 Fixed memory leak
- PR #123 Fixed sort predicates for nulls
- PR #116 Consistent split nulls
- PR #91 Caller allocs for to_host
- PR #87 Gather throws exception if args out of range
- PR #85 Fix join with null-string issue
- PR #43 cat values are ints
- PR #38 Catbitmask
- PR #37 use iterator instead of at() in regex eval
- PR #31 Change NVCategory values type from unsigned to signed int32
- PR #11 handle unescaped end-brace edge-case
- PR #157 Fix Python not being found by CMake
