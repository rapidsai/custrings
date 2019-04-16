
import pyniNVText
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
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["hello world",
    ...                          "goodbye world",
    ...                          "hello goodbye"])
    >>> ut = nvtext.unique_tokens(s)
    >>> print(ut)
    ["goodbye","hello","world"]

    """
    rtn = pyniNVText.n_unique_tokens(strs, delimiter)
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
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["hello world","goodbye",""])
    >>> n = nvtext.token_count(s)
    >>> print(n)
    [2,1,0]

    """
    rtn = pyniNVText.n_token_count(strs, delimiter, devptr)
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
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["hello","goodbye",""])
    >>> t = nvstrings.to_device(['o','y'])
    >>> n = nvtext.contains_strings(s,t)
    >>> print(n)
    [[True,False],[True,True],[False,False]]

    """
    rtn = pyniNVText.n_contains_strings(strs, tgts, devptr)
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
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["hello","goodbye",""])
    >>> t = nvstrings.to_device(['o','y'])
    >>> n = nvtext.strings_counts(s,t)
    >>> print(n)
    [[1,0],[2,1],[0,0]]

    """
    rtn = pyniNVText.n_strings_counts(strs, tgts, devptr)
    return rtn


def edit_distance(strs, tgt, algo=0, devptr=0):
    """
    Compute the edit-distance between strs and tgt.
    Edit distance is how many character changes between strings.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation.
    tgt : str, nvstrings
        The string or strings to compute edit-distance with.
    algo: int
        0 = Levenshtein
    devptr : GPU memory pointer
        Must be able to hold at least strs.size() of int32 values.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["honda","hyundai"])
    >>> n = nvtext.edit_distance(s,"honda")
    >>> print(n)
    [0,3]

    """
    rtn = pyniNVText.n_edit_distance(strs, tgt, algo, devptr)
    return rtn


def ngrams(strs, N=2, sep='_'):
    """Generate the n-grams of an nvstrings array.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation.
    N : int
        The degree of the n-gram (number of consecutive tokens). Default of 2
        for bigrams.
    sep : The separator to use between within an n-gram. Default of '_'.

    Returns
    -------
    ngrams_object : nvstrings

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> dstrings = nvstrings.to_device(['this is my', 'favorite book'])
    >>> print(nvtext.ngrams(dstrings, N=2, sep='_'))
    ['this_is', 'is_my', 'my_favorite', 'favorite_book']
    """
    # Tokenize
    tokens = strs.split_record()
    tokens_combined = nvs.from_strings(tokens)

    pad = nvs.to_device([''])
    ngram_object = tokens_combined
    total_num_of_tokens = tokens_combined.size()
    shifted_token_collection = []

    # Create shifted and padded nvstrings objects
    for i in range(N - 1):
        shifted_tokens = tokens_combined.remove_strings(
            list(range(0, i + 1))
        )
        shifted_tokens = shifted_tokens.add_strings(
            [pad] * (total_num_of_tokens - shifted_tokens.size())
        )
        shifted_token_collection.append(shifted_tokens)

    # Create the n-grams from the shifted nvstrings
    for sequence in shifted_token_collection:
        ngram_object = ngram_object.cat(sequence, sep)

    ngram_object = ngram_object.remove_strings(
        list(range(ngram_object.size() - N + 1, ngram_object.size()))
    )
    return ngram_object
