# cuStrings/nvStrings 0.9.0 (TBD)

- PR #355 Add and use unified build script

## New Features

- PR #363 Added scatter method to nvstrings
- PR #371 Added replace() with multiple strings parameters
- PR #385 Added Porter Stemmer measure method

## Improvements

- PR #354 Removed rmm submodule
- PR #359 Set default sort-type to alphabetical
- PR #357 Improved error checking on memory allocates
- PR #358 Added improved hash algorithm to reduce collisions
- PR #369 Reconfigure C++ source directory structure
- PR #370 Updated memsize() to include pointer memory too
- PR #375 Added gtests for C++ unit testing

## Bug Fixes

- PR #353 Fixed sizer calculation for multiple replaces
- PR #378 Fixed small memory leak in string categories
- PR #373 Properly catch/throw exceptions raised by `gather_strings`


# cuStrings/nvStrings 0.8.0 (27 June 2019)

## New Features

- PR #298 Added ngrams method to NVText
- PR #299 Added gather method for booleans array
- PR #304 Updated nvtext.ngrams to call C++ function
- PR #325 Add test skipping functionality to build.sh
- PR #332 Accept multiple nvstrings in cat() method
- PR #327 Allow numeric keys in nvcategory

## Improvements

- PR #316 Add .0 to integer output when using ftos/dtos
- PR #315 Support NaN, Inf strings in stof and stod
- PR #313 Increased regex instruction threshold for stack to 1024
- PR #302 Add junit test output for py.test
- PR #331 Use stack-size ranges for different regex instruction counts
- PR #341 Support device memory in from_offsets

## Bug Fixes

- PR #314 Fixed calculation error in ip2int
- PR #306 Set main label always for libnvstrings
- PR #307 Update conda dependencies
- PR #326 Update python doc version
- PR #335 to_host no longer overwrites input pointers for null strings
- PR #329 Fixed documentation errors in source code
- PR #334 Regex stack-size logic moved to host code
- PR #343 Fixed a flag which is necessary for conda install in Dockerfile
- PR #347 Configure Sphinx to render params correctly


# cuStrings/nvStrings 0.7.0 (10 May 2019)

Version jump from 0.3->0.7 is to align with other RAPIDS projects.

## New Features

- PR #295 Added nvtext.tokenize
- PR #281 Added get_info method to show data about strings
- PR #273 Added tokens_counts method to nvtext
- PR #265 Added is_empty method
- PR #260 Support for format-specified date-time to/from string conversion
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
- PR #285 Add build script for nightly docs
- PR #286 Add local build script to cuStrings


## Improvements

- PR #289 Improve regex perf for OR clauses
- PR #283 Check for empty fillchar in pad methods
- PR #278 Added insert method as alternative to slice_replace insert
- PR #268 Support negative slicing
- PR #264 Allow nvstrings parameter to fillna
- PR #256 Convert device-only custring_view to header-only
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
- PR #247 Release Python GIl while calling underlying C++ API from python
- PR #240 Initial suite of nvtext function Python unit tests
- PR #275 Add cudatoolkit conda dependency
- PR #291 Use latest release version in update-version CI script
- PR #267 Add a minimal suite of nvstrings and nvcategory pytests

## Bug Fixes

- PR #276 Fixed token_count counting empty string as token
- PR #269 Fixed exception on invalid parameter to edit_distance
- PR #261 Fixed doxygen formatting and updated to 0.7
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
- PR #274 Updated RPATH/RUNPATH setting to accommodate current install location


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
