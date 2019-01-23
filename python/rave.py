
import pyniRave
import nvstrings as nvs


def unique_tokens(strs, delimiter=' '):
    """
    Each string is split into tokens using the provided delimiter.
    The nvstrings instance returned contains unique list of tokens.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation

    delimiter : str
        The character used to locate the split points of each string.
        Default is space.

    Examples
    --------
    .. code-block:: python

    import nvstrings, rave

    s = nvstrings.to_device(["hello world","goodbye world","hello goodbye"])
    ut = rave.unique_tokens(s)
    print(ut)

    Output:

    .. code-block:: python

    ["goodbye","hello","world"]

    """
    rtn = pyniRave.n_unique_tokens(strs, delimiter)
    if rtn is not None:
        rtn = nvs.nvstrings(rtn)
    return rtn


def token_count(strs, delimiter=' ', devptr=0):
    """
    Each string is split into tokens using the provided delimiter.
    The returned integer array is the number of tokens in each string.

    Parameters
    ----------
        strs : nvstrings
            The strings for this operation

        delimiter : str
            The character used to locate the split points of each string.
            Default is space.

        devptr : GPU memory pointer
            Must be able to hold at least strs.size() of int32 values.

    Examples
    --------
    .. code-block:: python
      import nvstrings, rave

      s = nvstrings.to_device(["hello world","goodbye",""])
      n = rave.token_count(s)
      print(n)


    Output:

    .. code-block:: python
      [2,1,0]

    """
    rtn = pyniRave.n_token_count(strs, delimiter, devptr)
    return rtn


def contains_strings(strs, tgts, devptr=0):
    """
    The tgts strings are searched for within each strs.
    The returned byte array is 1 for each tgts in strs and 0 otherwise.

    Parameters
    ----------
        strs : nvstrings
            The strings for this operation.

        tgts : nvstrings
            The strings to check for inside each strs.

        devptr : GPU memory pointer
            Must be able to hold at least strs.size()*tgts.size()
            of int32 values.

    Examples
    --------
    .. code-block:: python

      import nvstrings, rave

      s = nvstrings.to_device(["hello","goodbye",""])
      t = nvstrings.to_device(['o','y'])
      n = rave.contains_strings(s,t)
      print(n)

    Output:

    .. code-block:: python
      [[True,False],[True,True],[False,False]]

    """
    rtn = pyniRave.n_contains_strings(strs, tgts, devptr)
    return rtn


def strings_counts(strs, tgts, devptr=0):
    """
    The tgts strings are searched for within each strs.
    The returned int32 array is number of occurrences of each tgts in strs.

    Parameters
    ----------
        strs : nvstrings
            The strings for this operation.

        tgts : nvstrings
            The strings to count for inside each strs.

        devptr : GPU memory pointer
            Must be able to hold at least strs.size()*tgts.size()
            of int32 values.

    Examples
    --------
    .. code-block:: python

      import nvstrings, rave

      s = nvstrings.to_device(["hello","goodbye",""])
      t = nvstrings.to_device(['o','y'])
      n = rave.strings_counts(s,t)
      print(n)


    Output:
    .. code-block:: python

      [[1,0],[2,1],[0,0]]

    """
    rtn = pyniRave.n_strings_counts(strs, tgts, devptr)
    return rtn
