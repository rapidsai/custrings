
import pyniNVCategory
import nvstrings as nvs


def to_device(strs):
    """
    Create a nvcategory object from a list of Python strings.

    Parameters
    ----------

      strs: list
        List of Python strings.

    Examples
    --------

    .. code-block:: python

    import nvcategory

    c = nvcategory.to_device(['apple','pear','banana','orange','pear'])
    print(c.keys(),c.values())

    Output:

    .. code-block:: python

    ['apple', 'banana', 'orange', 'pear'] [0, 3, 1, 2, 3]

    """
    rtn = pyniNVCategory.n_createCategoryFromHostStrings(strs)
    if rtn is not None:
        rtn = nvcategory(rtn)
    return rtn


def from_offsets(sbuf, obuf, scount, nbuf=None, ncount=0):
    """
    Create nvcategory object from byte-array of characters encoded in UTF-8.

    Parameters
    ----------

      sbuf : CPU memory address or buffer
        Strings characters encoded as UTF-8.

      obuf : CPU memory address or buffer
        Array of int32 byte offsets to beginning of each string in sbuf.
        There should be scount+1 values where the last value is the
        number of bytes in sbuf.

      scount: int
        Number of strings.

      nbuf: CPU memory address or buffer
        Optional null bitmask in arrow format.
        Strings with no lengths are empty strings unless specified as
        null by this bitmask.

      ncount: int
        Optional number of null strings.

      Examples
      --------

      .. code-block:: python

      import numpy as np
      import nvcategory

      # 'a','p','p','l','e' are utf8 int8 values 97,112,112,108,101
      values = np.array([97, 112, 112, 108, 101], dtype=np.int8)
      print("values",values.tobytes())
      offsets = np.array([0,1,2,3,4,5], dtype=np.int32)
      print("offsets",offsets)
      c = nvcategory.from_offsets(values,offsets,5)
      print(c.keys(),c.values())

      Output:

      .. code-block:: python

      values b'apple'
      offsets [0 1 2 3 4 5]
      ['a', 'e', 'l', 'p'] [0, 3, 3, 2, 1]

    """
    rtn = pyniNVCategory.n_createFromOffsets(sbuf, obuf, scount, nbuf, ncount)
    if rtn is not None:
        rtn = nvcategory(rtn)
    return rtn


def from_strings(*args):
    """
    Create a nvcategory object from a nvstrings object.

    Parameters
    ----------

      args: variadic
        1 or more nvstrings objects

    Examples
    --------

    .. code-block:: python

      import nvcategory, nvstrings

      s1 = nvstrings.to_device(['apple','pear','banana'])
      s2 = nvstrings.to_device(['orange','pear'])
      c = nvcategory.from_strings(s1,s2)
      print(c.keys(),c.values())

      Output:

      .. code-block:: python

      ['apple', 'banana', 'orange', 'pear'] [0, 3, 1, 2, 3]

    """
    strs = []
    for arg in args:
        strs.append(arg)
    rtn = pyniNVCategory.n_createCategoryFromNVStrings(strs)
    if rtn is not None:
        rtn = nvcategory(rtn)
    return rtn


def from_strings_list(list):
    """
    Create a nvcategory object from a list of nvstrings.

    Parameters
    ----------

      list: list
        1 or more nvstrings objects

    Examples
    --------

    .. code-block:: python

      import nvcategory, nvstrings

      s1 = nvstrings.to_device(['apple','pear','banana'])
      s2 = nvstrings.to_device(['orange','pear'])
      c = nvcategory.from_strings_list([s1,s2])
      print(c.keys(),c.values())

      Output:

      .. code-block:: python

      ['apple', 'banana', 'orange', 'pear'] [0, 3, 1, 2, 3]

    """
    rtn = pyniNVCategory.n_createCategoryFromNVStrings(list)
    if rtn is not None:
        rtn = nvcategory(rtn)
    return rtn


class nvcategory:
    """
    Instance manages a dictionary of strings (keys) in device memory
    and a mapping of indexes (values).

    """
    #
    m_cptr = 0

    def __init__(self, cptr):
        """For internal use only."""
        self.m_cptr = cptr

    def __del__(self):
        pyniNVCategory.n_destroyCategory(self.m_cptr)

    def __str__(self):
        return str(self.keys())

    def __repr__(self):
        return "<nvcategory keys={},values={}>".format(
                self.keys_size(), self.size())

    def size(self):
        """
        The number of values.

        Returns
        -------
          int: number of values

        Examples
        --------

        .. code-block:: python

          import nvcategory
          c = nvcategory.to_device(["eee","aaa","eee","dddd"])
          print(c.values())
          print(c.size())

        Output:

        .. code-block:: python

          [2, 0, 2, 1]
          4

        """
        return pyniNVCategory.n_size(self.m_cptr)

    def keys_size(self):
        """
        The number of keys.

        Returns
        -------
          int: number of keys

        Examples
        --------

        .. code-block:: python

          import nvcategory
          c = nvcategory.to_device(["eee","aaa","eee","dddd"])
          print(c.keys())
          print(c.keys_size())

        Output:

        .. code-block:: python

          ['aaa','dddd','eee']
          3

        """
        return pyniNVCategory.n_keys_size(self.m_cptr)

    def keys(self):
        """
        Return the unique strings for this category as nvstrings instance.

        Returns
        -------
          nvstrings: keys

        Examples
        --------

        .. code-block:: python

          import nvcategory
          c = nvcategory.to_device(["eee","aaa","eee","dddd"])
          print(c.keys())

        Output:

        .. code-block:: python

          ['aaa','dddd','eee']

        """
        rtn = pyniNVCategory.n_get_keys(self.m_cptr)
        if rtn is not None:
            rtn = nvs.nvstrings(rtn)
        return rtn

    def indexes_for_key(self, key, devptr=0):
        """
        Return all index values for given key.

        Parameters
        ----------
          key : str
            key whose values should be returned

          devptr : GPU memory pointer
            Where index values will be written.
            Must be able to hold int32 values for this key.

        Examples
        --------

        .. code-block:: python

          import nvcategory
          c = nvcategory.to_device(["eee","aaa","eee","dddd"])
          print(c.indexes_for_key('aaa'))
          print(c.indexes_for_key('eee'))

        Output:

        .. code-block:: python

          [1]
          [0, 2]

        """
        return pyniNVCategory.n_get_indexes_for_key(self.m_cptr, key, devptr)

    def value_for_index(self, idx):
        """
        Return the category value for the given index.

        Parameters
        ----------
          idx : int
            index value to retrieve

        Examples
        --------

        .. code-block:: python

          import nvcategory
          c = nvcategory.to_device(["eee","aaa","eee","dddd"])
          print(c.value_for_index(3))

        Output:

        .. code-block:: python

          1

        """
        return pyniNVCategory.n_get_value_for_index(self.m_cptr, idx)

    def value(self, str):
        """
        Return the category value for the given string.

        Parameters
        ----------
          str : str
            key to retrieve

        Examples
        --------

        .. code-block:: python

          import nvcategory
          c = nvcategory.to_device(["eee","aaa","eee","dddd"])
          print(c.value('aaa'))
          print(c.value('eee'))

        Output:

        .. code-block:: python

          0
          2

        """
        return pyniNVCategory.n_get_value_for_string(self.m_cptr, str)

    def values(self, devptr=0):
        """
        Return all values for this instance.

        Parameters
        ----------
          devptr : GPU memory pointer
            Where index values will be written.
            Must be able to hold size() of int32 values.

        Examples
        --------

        .. code-block:: python

          import nvcategory
          c = nvcategory.to_device(["eee","aaa","eee","dddd"])
          print(c.values())

        Output:

        .. code-block:: python

          [2, 0, 2, 1]

        """
        return pyniNVCategory.n_get_values(self.m_cptr, devptr)

    def add_strings(self, nvs):
        """
        Create new category incorporating specified strings.
        This will return a new nvcategory with new key values.
        The index values will appear as if appended.

        Parameters
        ----------
          nvs : nvstrings
            New strings to be added.

        Examples
        --------

        .. code-block:: python

          import nvcategory, nvstrings
          s1 = nvstrings.to_device(["eee","aaa","eee","dddd"])
          s2 = nvstrings.to_device(["ggg","eee","aaa"])
          c1 = nvcategory.from_strings(s1)
          c2 = c1.add_strings(s2)
          print(c1.keys())
          print(c1.values())
          print(c2.keys())
          print(c2.values())

        Output:

        .. code-block:: python
          ['aaa','dddd','eee']
          [2, 0, 2, 1]
          ['aaa','dddd','eee','ggg']
          [2, 0, 2, 1, 3, 2, 0]

        """
        rtn = pyniNVCategory.n_add_strings(self.m_cptr, nvs)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn

    def remove_strings(self, nvs):
        """
        Create new category without the specified strings.
        The returned category will have new set of key values and indexes.

        Parameters
        ----------
          nvs : nvstrings
            strings to be removed.

        Examples
        --------

        .. code-block:: python

          import nvcategory, nvstrings
          s1 = nvstrings.to_device(["eee","aaa","eee","dddd"])
          s2 = nvstrings.to_device(["aaa"])
          c1 = nvcategory.from_strings(s1)
          c2 = c1.remove_strings(s2)
          print(c1.keys())
          print(c1.values())
          print(c2.keys())
          print(c2.values())

        Output:

        .. code-block:: python
          ['aaa','dddd','eee']
          [2, 0, 2, 1]
          ['dddd', 'eee']
          [1, 1, 0]

        """
        rtn = pyniNVCategory.n_remove_strings(self.m_cptr, nvs)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn

    def to_strings(self):
        """
        Return nvstrings instance represented by the values in this instance.

        Returns
        -------
          nvstrings: full strings list based on values indexes

        Examples
        --------

        .. code-block:: python

          import nvcategory
          c = nvcategory.to_device(["eee","aaa","eee","dddd"])
          print(c.keys())
          print(c.values())
          print(c.to_strings())

        Output:

        .. code-block:: python

          ['aaa','dddd','eee']
          [2, 0, 2, 1]
          ['eee','aaa','eee','dddd']

        """
        rtn = pyniNVCategory.n_to_strings(self.m_cptr)
        if rtn is not None:
            rtn = nvs.nvstrings(rtn)
        return rtn

    def gather_strings(self, indexes, count=0):
        """
        Return nvstrings instance represented using the specified indexes.

        Parameters
        ----------
          indexes : List of ints or GPU memory pointer
            0-based indexes of keys to return as an nvstrings object

          count : int
            Number of ints if indexes parm is a device pointer.
            Otherwise it is ignored.

        Returns
        -------
          nvstrings: strings list based on indexes

        Examples
        --------

        .. code-block:: python

          import nvcategory
          c = nvcategory.to_device(["eee","aaa","eee","dddd"])
          print(c.keys())
          print(c.values())
          print(c.gather_strings([0,2,0]))

        Output:

        .. code-block:: python

          ['aaa','dddd','eee']
          [2, 0, 2, 1]
          ['aaa','eee','aaa']

        """
        rtn = pyniNVCategory.n_gather_strings(self.m_cptr, indexes, count)
        if rtn is not None:
            rtn = nvs.nvstrings(rtn)
        return rtn

    def merge_category(self, nvcat):
        """
        Create new category incorporating the specified category keys
        and values. This will return a new nvcategory with new key values.
        The index values will appear as if appended. Any matching keys
        will preserve their values and any new keys will get new values.

        Parameters
        ----------
          nvcat : nvcategory
            New cateogry to be merged.

        """
        rtn = pyniNVCategory.n_merge_category(self.m_cptr, nvcat)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn

    def merge_and_remap(self, nvcat):
        """
        Create new category incorporating the specified category keys
        and values. This will return a new nvcategory with new key values.
        The index values will appear as if appended.
        Values will be remapped to the new keys.

        Parameters
        ----------
          nvcat : nvcategory
            New cateogry to be merged.

        """
        rtn = pyniNVCategory.n_merge_and_remap(self.m_cptr, nvcat)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn

    def add_keys(self, strs):
        """
        Create new category adding the specified keys and remapping
        values to the new key indexes.

        Parameters
        ----------
          strs: nvstrings
            keys to be added to existing keys
        """
        rtn = pyniNVCategory.n_add_keys(self.m_cptr, strs)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn

    def remove_keys(self, strs):
        """
        Create new category removing the specified keys and remapping
        values to the new key indexes. Values with removed keys are
        mapped to -1.

        Parameters
        ----------
          strs: nvstrings
            keys to be removed from existing keys
        """
        rtn = pyniNVCategory.n_remove_keys(self.m_cptr, strs)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn

    def set_keys(self, strs):
        """
        Create new category using the specified keys and remapping
        values to the new key indexes. Matching names will have
        remapped values. Values with removed keys are mapped to -1.

        Parameters
        ----------
          strs: nvstrings
            keys to be used for new category
        """
        rtn = pyniNVCategory.n_set_keys(self.m_cptr, strs)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn
