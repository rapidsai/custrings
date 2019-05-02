# Contributing to cuStrings

If you are interested in contributing to cuStrings, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/rapidsai/custrings/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - The RAPIDS team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    provide.

## Code contributions

### Your first issue

1. Follow the guide at the bottom of this page for [Setting Up Your Build Environment](#setting-up-your-build-environment)
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/rapidsai/custrings/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/rapidsai/custrings/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels
3. Comment on the issue saying you are going to work on it
4. Code! Make sure to update unit tests!
5. When done, [create your pull request](https://github.com/rapidsai/custrings/compare)
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed
7. Wait for other developers to review your code and update code as needed
8. Once reviewed and approved, a RAPIDS developer will merge your pull request

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues of our next release in our [project boards](https://github.com/rapidsai/custrings/projects).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where RAPIDS developers also focus their efforts.

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

### Building and Testing on a gpuCI image locally

Before submitting a pull request, you can do a local build and test on your machine that mimics our gpuCI environment using the `ci/local/build.sh` script.
For detailed information on usage of this script, see [here](ci/local/README.md).

## Setting Up Your Build Environment

The following instructions are for developers and contributors to cuStrings OSS development. These instructions are tested on Linux Ubuntu 16.04 & 18.04. Use these instructions to build cuStrings from source and contribute to its development.  Other operatings systems may be compatible, but are not currently tested.

See the [C++ readme](cpp/README.md) for instructions building and installing.

The nvStrings library has no dependencies on cuDF.

## Project Structure Overview

### cpp

C/C++ source code including python bindings.
See [/cpp/README.md](cpp/README.md)

### python

Python modules and wrappers including test scripts.
See [/python/README.md](python/README.md)

### data

These are sample data files for testing the library features.
There is no source code here but this is considered an active directory since many of the test cases rely on them.

### docs

Documentation of python interfaces generated from the python source code.
This also includes some general documentation on capabilities and limitations.

### conda

Support files for deploying to conda environments.
