import pyniNVStrings


def to_device(strs):
    """
    Create nvstrings instance from list of Python strings.

    Parameters
    ----------

      strs: list
        List of Python strings.

    Examples
    --------

    .. code-block:: python

    import nvstrings

    s = nvstrings.to_device(['apple','pear','banana','orange'])
    print(s)

    Output:

    .. code-block:: python

    ['apple', 'pear', 'banana', 'orange']


    """
    rtn = pyniNVStrings.n_createFromHostStrings(strs)
    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def from_csv(csv, column, lines=0, flags=0):
    """
    Reads a column of values from a CSV file into a new nvstrings instance.
    The CSV file must be formatted as UTF-8.

    Parameters
    ----------
        csv : str
            Path to the csv file from which to load data
        column : int
            0-based index of the column to read into an nvstrings object
        lines : int
            maximum number of lines to read from the file
        flags : int
            values may be combined
            1 - sort by length
            2 - sort by name
            8 - nulls are empty strings

    Returns
    -------
    A new nvstrings instance pointing to strings loaded onto the GPU

    Examples
    --------
    For CSV file (file.csv) containing 2 rows and 3 columns:
    header1,header2,header3
    r1c1,r1c2,r1c3
    r2c1,r2c2,r2c3

    .. code-block:: python

      import nvstrings
      s = nvstrings.from_csv("file.csv",2)
      print(s)

    Output:

    .. code-block:: python

      ['r1c3','r2c3']

    """
    rtn = pyniNVStrings.n_createFromCSV(csv, column, lines, flags)
    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def from_offsets(sbuf, obuf, scount, nbuf=None, ncount=0):
    """
    Create nvstrings object from byte-array of characters encoded in UTF-8.

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
      import nvstrings

      # 'a','p','p','l','e' are utf8 int8 values 97,112,112,108,101
      values = np.array([97, 112, 112, 108, 101], dtype=np.int8)
      print("values",values.tobytes())
      offsets = np.array([0,1,2,3,4,5], dtype=np.int32)
      print("offsets",offsets)
      s = nvstrings.from_offsets(values,offsets,5)
      print(s)

      Output:

      .. code-block:: python

      values b'apple'
      offsets [0 1 2 3 4 5]
      ['a', 'p', 'p', 'l', 'e']

    """
    rtn = pyniNVStrings.n_createFromOffsets(sbuf, obuf, scount, nbuf, ncount)
    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def free(dstrs):
    """Force free resources for the specified instance."""
    if dstrs is not None:
        pyniNVStrings.n_destroyStrings(dstrs.m_cptr)
        dstrs.m_cptr = 0


def bind_cpointer(cptr):
    """Bind an NVStrings C-pointer to a new instance."""
    rtn = None
    if cptr != 0:
        rtn = nvstrings(cptr)
    return rtn


# this will be documented with all the public methods
class nvstrings:
    """
    Instance manages a list of strings in device memory.

    Operations are across all of the strings and their results reside in
    device memory. Strings in the list are immutable.
    Methods that modify any string will create a new nvstrings instance.
    """
    #
    m_cptr = 0

    def __init__(self, cptr):
        """
        Use to_device() to create new instance from Python array of strings.
        """
        self.m_cptr = cptr

    def __del__(self):
        pyniNVStrings.n_destroyStrings(self.m_cptr)
        self.m_cptr = 0

    def __str__(self):
        return str(pyniNVStrings.n_createHostStrings(self.m_cptr))

    def __repr__(self):
        return "<nvstrings count={}>".format(self.size())

    def __getitem__(self, key):
        """
        Implemented for [] operator on nvstrings.
        Parameter must be integer, slice, or list of integers.
        """
        if key is None:
            raise KeyError("key must not be None")
        if isinstance(key, list):
            return self.gather(key)
        if isinstance(key, int):
            return self.gather([key])
        if isinstance(key, slice):
            start = 0 if key.start is None else key.start
            end = self.size() if key.stop is None else key.stop
            step = 1 if key.step is None or key.step is 0 else key.step
            rtn = pyniNVStrings.n_sublist(self.m_cptr, start, end, step)
            if rtn is not None:
                rtn = nvstrings(rtn)
            return rtn
        # raise TypeError("key must be integer, slice, or list of integers")
        # gather can handle almost anything now
        return self.gather(key)

    def __iter__(self):
        raise TypeError("iterable not supported by nvstrings")

    def __len__(self):
        return self.size()

    def to_host(self):
        """
        Copies strings back to CPU memory into a Python array.

        Returns
        -------
        A list of strings

        Examples
        --------

        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","world"])

          h = s.upper().to_host()
          print(h)

        Output:

        .. code-block:: python

          ["HELLO","WORLD"]

        """
        return pyniNVStrings.n_createHostStrings(self.m_cptr)

    def to_offsets(self, sbuf, obuf, nbuf=0, bdevmem=False):
        """
        Store byte-array of characters encoded in UTF-8 and offsets
        and optional null-bitmask into provided memory.

        Parameters
        ----------

          sbuf : memory address or buffer
            Strings characters are stored contiguously encoded as UTF-8.

          obuf : memory address or buffer
            Stores array of int32 byte offsets to beginning of each
            string in sbuf. This should be able to hold size()+1 values.

          nbuf: memory address or buffer
            Optional: stores null bitmask in arrow format.

          bdevmem: boolean
            Default (False) interprets memory pointers as CPU memory.

          Examples
          --------

          .. code-block:: python

          import numpy as np
          import nvstrings

          s = nvstrings.to_device(['a','p','p','l','e'])
          values = np.empty(s.size(), dtype=np.int8)
          offsets = np.empty(s.size()+1, dtype=np.int32)
          s.to_offsets(values,offsets)
          print("values",values.tobytes())
          print("offsets",offsets)

          Output:

          .. code-block:: python

          values b'apple'
          offsets [0 1 2 3 4 5]

        """
        return pyniNVStrings.n_create_offsets(self.m_cptr, sbuf, obuf, nbuf,
                                              bdevmem)

    def size(self):
        """
        The number of strings managed by this instance.

        Returns
        -------
          int: number of strings

        Examples
        --------

        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","world"])
          print(s.size())

        Output:

        .. code-block:: python

          2

        """
        return pyniNVStrings.n_size(self.m_cptr)

    def len(self, devptr=0):
        """
        Returns the number of characters of each string.

        Parameters
        ----------
            devptr : GPU memory pointer
                Where string length values will be written.
                Must be able to hold at least size() of int32 values.

        Examples
        --------

        .. code-block:: python

          import nvstrings
          import numpy as np
          from librmm_cffi import librmm

          # example passing device memory pointer
          s = nvstrings.to_device(["abc","d","ef"])
          arr = np.arange(s.size(),dtype=np.int32)
          d_arr = librmm.to_device(arr)
          s.len(d_arr.device_ctypes_pointer.value)
          print(d_arr.copy_to_host())

        Output:

        .. code-block:: python

          [3,1,2]

        """
        rtn = pyniNVStrings.n_len(self.m_cptr, devptr)
        return rtn

    def byte_count(self, vals=0, bdevmem=False):
        """
        Fills the argument with the number of bytes of each string.
        Returns the total number of bytes.

        Parameters
        ----------
            vals : memory pointer
                Where byte length values will be written.
                Must be able to hold at least size() of int32 values.
                None can be specified if only the total count is required.

        Examples
        --------

        .. code-block:: python

          import nvstrings
          import numpy as np
          from librmm_cffi import librmm

          # example passing device memory pointer
          s = nvstrings.to_device(["abc","d","ef"])
          arr = np.arange(s.size(),dtype=np.int32)
          d_arr = librmm.to_device(arr)
          s.byte_count(d_arr.device_ctypes_pointer.value,True)
          print(d_arr.copy_to_host())

        Output:

        .. code-block:: python

          [3,1,2]

        """
        rtn = pyniNVStrings.n_byte_count(self.m_cptr, vals, bdevmem)
        return rtn

    def compare(self, str, devptr=0):
        """
        Compare each string to the supplied string.
        Returns value of 0 for strings that match str.
        Returns < 0 when first different character is lower
        than argument string or argument string is shorter.
        Returns > 0 when first different character is greater
        than the argument string or the argument string is longer.

        Parameters
        ----------
            str : str
                String to compare all strings in this instance.

            devptr : GPU memory pointer
                Where string result values will be written.
                Must be able to hold at least size() of int32 values.

        Examples
        --------

        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","world"])

          print(s.compare('hello'))

        Output:

        .. code-block:: python

          [0,15]

        """
        rtn = pyniNVStrings.n_compare(self.m_cptr, str, devptr)
        return rtn

    def hash(self, devptr=0):
        """
        Returns hash values represented by each string.

        Parameters
        ----------
            devptr : GPU memory pointer
                Where string hash values will be written.
                Must be able to hold at least size() of uint32 values.

        Examples
        --------

        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","world"])
          s.hash()

        Output:

        .. code-block:: python

          [99162322, 113318802]

        """
        rtn = pyniNVStrings.n_hash(self.m_cptr, devptr)
        return rtn

    def stoi(self, devptr=0):
        """
        Returns integer value represented by each string.

        Parameters
        ----------
            devptr : GPU memory pointer
                Where resulting integer values will be written.
                Memory must be able to hold at least size() of int32 values.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["1234","-876","543.2","-0.12",".55""])
          print(s.stoi())

        Output:

        .. code-block:: python

          [1234, -876, 543, 0, 0]

        """
        rtn = pyniNVStrings.n_stoi(self.m_cptr, devptr)
        return rtn

    def stof(self, devptr=0):
        """
        Returns float values represented by each string.

        Parameters
        ----------
            devptr : GPU memory pointer
                Where resulting float values will be written.
                Memory must be able to hold at least size() of float32 values

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["1234","-876","543.2","-0.12",".55"])
          print(s.stof())

        Output:

        .. code-block:: python

          [1234.0, -876.0, 543.2000122070312,
           -0.11999999731779099, 0.550000011920929]

        """
        rtn = pyniNVStrings.n_stof(self.m_cptr, devptr)
        return rtn

    def htoi(self, devptr=0):
        """
        Returns integer value represented by each string.
        String is interpretted to have hex (base-16) characters.

        Parameters
        ----------
            devptr : GPU memory pointer
                Where resulting integer values will be written.
                Memory must be able to hold at least size() of int32 values.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["1234","ABCDEF","1A2","cafe"])
          print(s.htoi())

        Output:

        .. code-block:: python

          [4660, 11259375, 418, 51966]

        """
        rtn = pyniNVStrings.n_htoi(self.m_cptr, devptr)
        return rtn

    def cat(self, others=None, sep=None, na_rep=None):
        """
        Appends the given strings to this list of strings and
        returns as new nvstrings.

        Parameters
        ----------
            others : List of str
                Strings to be appended.
                The number of strings must match size() of this instance.
                This must be either a Python array of strings or another
                nvstrings instance.

            sep : str
                If specified, this separator will be appended to each string
                before appending the others.

            na_rep : char
                This character will take the place of any null strings
                (not empty strings) in either list.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s1 = nvstrings.to_device(['hello', None,'goodbye'])
          s2 = nvstrings.to_device(['world','globe', None])

          print(s1.cat(s2,sep=':', na_rep='_'))

        Output:

        .. code-block:: python

          ["hello:world","_:globe","goodbye:_"]

        """
        rtn = pyniNVStrings.n_cat(self.m_cptr, others, sep, na_rep)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def join(self, sep=''):
        """
        Concatentate this list of strings into a single string.

        Parameters
        ----------
            sep : str
                This separator will be appended to each string before
                appending the next.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye"])
          s.join(sep=':')

        Output:

        .. code-block:: python

          ['hello:goodbye']

        """
        rtn = pyniNVStrings.n_join(self.m_cptr, sep)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def split(self, delimiter=None, n=-1):
        """
        Returns an array of nvstrings each representing the split
        of each individual string.

        Parameters
        ----------
            delimiter : str
                The character used to locate the split points of
                each string. Default is space.

            n : int
                Maximum number of strings to return for each split.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello world","goodbye","well said"])
          for result in s.split(' '):
            print(result)


        Output:

        .. code-block:: python

          ["hello","world"]
          ["goodbye"]
          ["well","said"]

        """
        strs = pyniNVStrings.n_split(self.m_cptr, delimiter, n)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def rsplit(self, delimiter=None, n=-1):
        """
        Returns an array of nvstrings each representing the split of each
        individual string. The delimiter is searched for from the end of
        each string.

        Parameters
        ----------
            delimiter : str
                The character used to locate the split points of each
                string. Default is space.

            n : int
                Maximum number of strings to return for each split.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          strs = nvstrings.to_device(["hello world","goodbye","up in arms"])
          for s in strs.rsplit(' ',2):
            print(s)


        Output:

        .. code-block:: python

          ['hello', 'world']
          ['goodbye']
          ['up in', 'arms']

        """
        strs = pyniNVStrings.n_rsplit(self.m_cptr, delimiter, n)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def partition(self, delimiter=' '):
        """
        Each string is split into two strings on the first delimiter found.

        Three strings are returned for each string:
        beginning, delimiter, end.

        Parameters
        ----------
            delimiter : str
                The character used to locate the split points of each
                string. Default is space.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          strs = nvstrings.to_device(["hello world","goodbye","up in arms"])
          for s in strs.partition(' '):
            print(s)


        Output:

        .. code-block:: python

          ['hello', ' ', 'world']
          ['goodbye', '', '']
          ['up', ' ', 'in arms']

        """
        strs = pyniNVStrings.n_partition(self.m_cptr, delimiter)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def rpartition(self, delimiter=' '):
        """
        Each string is split into two strings on the first delimiter found.
        Delimiter is searched for from the end.

        Three strings are returned for each string: beginning, delimiter, end.

        Parameters
        ----------
            delimiter : str
                The character used to locate the split points of each string.
                Default is space.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          strs = nvstrings.to_device(["hello world","goodbye","up in arms"])
          for s in strs.rpartition(' '):
            print(s)


        Output:

        .. code-block:: python

          ['hello', ' ', 'world']
          ['', '', 'goodbye']
          ['up in', ' ', 'arms']

        """
        strs = pyniNVStrings.n_rpartition(self.m_cptr, delimiter)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def split_column(self, delimiter=' ', n=-1):
        """
        A new set of columns (nvstrings) is created by splitting
        the strings vertically.

        Parameters
        ----------
            delimiter : str
                The character used to locate the split points of each string.
                Default is space.

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello world","goodbye","well said"])
          for result in s.split_column(' '):
            print(result)


        Output:

        .. code-block:: python

          ["hello","goodbye","well"]
          ["world",None,"said"]

        """
        strs = pyniNVStrings.n_split_column(self.m_cptr, delimiter, n)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def rsplit_column(self, delimiter=' ', n=-1):
        """
        A new set of columns (nvstrings) is created by splitting
        the strings vertically. Delimiter is searched from the end.

        Parameters
        ----------
            delimiter : str
                The character used to locate the split points of each string.
                Default is space.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello world","goodbye","well said"])
          for result in s.rsplit_column(' '):
            print(result)


        Output:

        .. code-block:: python

          ["hello","goodbye","well"]
          ["world",None,"said"]

        """
        strs = pyniNVStrings.n_rsplit_column(self.m_cptr, delimiter, n)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def get(self, i):
        """
        Returns the character specified in each string as a new string.

        The nvstrings returned contains a list of single character strings.

        Parameters
        ----------
          i : int
            The character position identifying the character
            in each string to return.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello world","goodbye","well said"])
          print(s.get(0))

        Output:

        .. code-block:: python

          ['h', 'g', 'w']

        """
        rtn = pyniNVStrings.n_get(self.m_cptr, i)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def repeat(self, repeats):
        """
        Appends each string with itself the specified number of times.
        This returns a nvstrings instance with the new strings.

        Parameters
        ----------
            repeats : int
               The number of times each string should be repeated.
               Repeat count of 0 or 1 will just return copy of each string.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye","well"])
          print(s.repeat(2))

        Output:

        .. code-block:: python

          ['hellohello', 'goodbyegoodbye', 'wellwell']

        """
        rtn = pyniNVStrings.n_repeat(self.m_cptr, repeats)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def pad(self, width, side='left', fillchar=' '):
        """
        Add specified padding to each string.
        Side:{'left','right','both'}, default is 'left'.

        Parameters
        ----------
          fillchar : char
            The character used to do the padding.
            Default is space character. Only the first character is used.

          side : str
            Either one of "left", "right", "both". The default is "left"

            "left" performs a padding on the left – same as rjust()

            "right" performs a padding on the right – same as ljust()

            "both" performs equal padding on left and right
            – same as center()

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye","well"])
          print(s.pad(' ', side='left'))

        Output:

        .. code-block:: python

          [" hello"," goodbye"," well"]

        """
        rtn = pyniNVStrings.n_pad(self.m_cptr, width, side, fillchar)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def ljust(self, width, fillchar=' '):
        """
        Pad the end of each string to the minimum width.

        Parameters
        ----------
          width	: int
            The minimum width of characters of the new string.
            If the width is smaller than the existing string,
            no padding is performed.

          fillchar : char
            The character used to do the padding.
            Default is space character. Only the first character is used.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye","well"])
          print(s.ljust(width=6))

        Output:

        .. code-block:: python

          ['hello ', 'goodbye', 'well  ']

        """
        rtn = pyniNVStrings.n_ljust(self.m_cptr, width)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def center(self, width, fillchar=' '):
        """
        Pad the beginning and end of each string to the minimum width.

        Parameters
        ----------
          width	: int
            The minimum width of characters of the new string.
            If the width is smaller than the existing string,
            no padding is performed.

          fillchar : char
            The character used to do the padding.
            Default is space character. Only the first character is used.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye","well"])
          for result in s.center(width=6):
            print(result)


        Output:

        .. code-block:: python

          ['hello ', 'goodbye', ' well ']

        """
        rtn = pyniNVStrings.n_center(self.m_cptr, width, fillchar)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def rjust(self, width, fillchar=' '):
        """
        Pad the beginning of each string to the minimum width.

        Parameters
        ----------
          width	: int
            The minimum width of characters of the new string.
            If the width is smaller than the existing string,
            no padding is performed.

          fillchar : char
            The character used to do the padding.
            Default is space character. Only the first character is used.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye","well"])
          print(s.ljust(width=6))

        Output:

        .. code-block:: python

          [' hello', 'goodbye', '  well']

        """
        rtn = pyniNVStrings.n_rjust(self.m_cptr, width)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def zfill(self, width):
        """
        Pads the strings with leading zeros.
        It will handle prefix sign characters correctly for strings
        containing leading number characters.

        Parameters
        ----------
          width	: int
            The minimum width of characters of the new string.
            If the width is smaller than the existing string,
            no padding is performed.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","1234","-9876","+5.34"])
          print(s.zfill(width=6))

        Output:

        .. code-block:: python

          ['0hello', '001234', '-09876', '+05.34']

        """
        rtn = pyniNVStrings.n_zfill(self.m_cptr, width)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def wrap(self, width):
        """
        This will place new-line characters in whitespace so each line
        is no more than width characters. Lines will not be truncated.

        Parameters
        ----------
          width	: int
            The maximum width of characters per newline in the new string.
            If the width is smaller than the existing string, no newlines
            will be inserted.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello there","goodbye all","well ok"])
          print(s.wrap(3))

        Output:

        .. code-block:: python

          ['hello\\nthere', 'goodbye\\nall', 'well\\nok']

        """
        rtn = pyniNVStrings.n_wrap(self.m_cptr, width)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def slice(self, start, stop=None, step=None):
        """
        Returns a substring of each string.

        Parameters
        ----------
        start : int
          Beginning position of the string to extract.
          Default is beginning of the each string.

        stop : int
          Ending position of the string to extract.
          Default is end of each string.

        step : str
          Characters that are to be captured within the specified section.
          Default is every character.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye"])
          print(s.slice(2,5))

        Output:

        .. code-block:: python

          ['llo', 'odb']

        """
        rtn = pyniNVStrings.n_slice(self.m_cptr, start, stop, step)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def slice_from(self, starts=0, stops=0):
        """
        Return substring of each string using positions for each string.

        The starts and stops parameters are device memory pointers.
        If specified, each must contain size() of int32 values.

        Parameters
        ----------
        starts : GPU memory pointer
          Beginning position of each the string to extract.
          Default is beginning of the each string.

        stops : GPU memory pointer
          Ending position of the each string to extract.
          Default is end of each string.
          Use -1 to specify to the end of that string.


        Examples
        --------
        .. code-block:: python

          import nvstrings
          import numpy as np
          from numba import cuda

          s = nvstrings.to_device(["hello","there"])
          darr = cuda.to_device(np.asarray([2,3],dtype=np.int32))
          print(s.slice_from(starts=darr.device_ctypes_pointer.value))

        Output:

        .. code-block:: python

          ['llo','re']

        """
        rtn = pyniNVStrings.n_slice_from(self.m_cptr, starts, stops)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def slice_replace(self, start=None, stop=None, repl=None):
        """
        Replace the specified section of each string with a new string.

        Parameters
        ----------
        start : int
          Beginning position of the string to replace.
          Default is beginning of the each string.

        stop : int
          Ending position of the string to replace.
          Default is end of each string.

        repl : str
          String to insert into the specified position values.

        Examples
        --------
        .. code-block:: python

        import nvstrings

        strs = nvstrings.to_device(["abcdefghij","0123456789"])
        print(strs.slice_replace(2,5,'z'))

        Output:

        .. code-block:: python

        ['abzfghij', '01z56789']

        """
        rtn = pyniNVStrings.n_slice_replace(self.m_cptr, start, stop, repl)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def replace(self, pat, repl, n=-1, regex=True):
        """
        Replace a string (pat) in each string with another string (repl).

        Parameters
        ----------
        pat : str
          String to be replaced.
          This can also be a regex expression -- not a compiled regex.

        repl : str
          String to replace `strng` with

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye"])
          print(s.replace('e', ''))

        Output:

        .. code-block:: python

          ['hllo', 'goodby']

        """
        rtn = pyniNVStrings.n_replace(self.m_cptr, pat, repl, n, regex)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def lstrip(self, to_strip=None):
        """
        Strip leading characters from each string.

        Parameters
        ----------
        to_strip : str
          Characters to be removed from leading edge of each string

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["oh","hello","goodbye"])
          print(s.lstrip('o'))

        Output:

        .. code-block:: python

          ['h', 'hello', 'goodbye']

        """
        rtn = pyniNVStrings.n_lstrip(self.m_cptr, to_strip)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def strip(self, to_strip=None):
        """
        Strip leading and trailing characters from each string.

        Parameters
        ----------
        to_strip : str
          Characters to be removed from both ends of each string

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["oh, hello","goodbye"])
          print(s.strip('o'))

        Output:

        .. code-block:: python

          ['h, hell', 'goodbye']

        """
        rtn = pyniNVStrings.n_strip(self.m_cptr, to_strip)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def rstrip(self, to_strip=None):
        """
        Strip trailing characters from each string.

        Parameters
        ----------
        to_strip : str
          Characters to be removed from trailing edge of each string

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["oh","hello","goodbye"])
          print(s.rstrip('o'))

        Output:

        .. code-block:: python

          ['oh', 'hell', 'goodbye']

        """
        rtn = pyniNVStrings.n_rstrip(self.m_cptr, to_strip)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def lower(self):
        """
        Convert each string to lowercase.
        This only applies to ASCII characters at this time.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["Hello, Friend","Goodbye, Friend"])
          print(s.lower())

        Output:

        .. code-block:: python

          ['hello, friend', 'goodbye, friend']

        """
        rtn = pyniNVStrings.n_lower(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def upper(self):
        """
        Convert each string to uppercase.
        This only applies to ASCII characters at this time.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["Hello, friend","Goodbye, friend"])
          print(s.lower())

        Output:

        .. code-block:: python

          ['HELLO, FRIEND', 'GOODBYE, FRIEND']

        """
        rtn = pyniNVStrings.n_upper(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def capitalize(self):
        """
        Capitalize first character of each string.
        This only applies to ASCII characters at this time.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello, friend","goodbye, friend"])
          print(s.lower())

        Output:

        .. code-block:: python

          ['Hello, friend", "Goodbye, friend"]

        """
        rtn = pyniNVStrings.n_capitalize(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def swapcase(self):
        """
        Change each lowercase character to uppercase and vice versa.
        This only applies to ASCII characters at this time.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["Hello, Friend","Goodbye, Friend"])
          print(s.lower())

        Output:

        .. code-block:: python

          ['hELLO, fRIEND', 'gOODBYE, fRIEND']

        """
        rtn = pyniNVStrings.n_swapcase(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def title(self):
        """
        Uppercase the first letter of each letter after a space
        and lowercase the rest.
        This only applies to ASCII characters at this time.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["Hello friend","goodnight moon"])
          print(s.title())

        Output:

        .. code-block:: python

          ['Hello Friend', 'Goodnight Moon']

        """
        rtn = pyniNVStrings.n_title(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def index(self, sub, start=0, end=None, devptr=0):
        """
        Same as find but throws an error if arg is not found in all strings.

        Parameters
        ----------
          sub : str
            String to find

          start : int
            Beginning of section to replace.
            Default is beginning of each string.

          end : int
            End of section to replace. Default is end of each string.

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory size must be able to hold at least size() of int32 values.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","world"])

          print(s.index('l'))

        Output:

        .. code-block:: python

          [2,3]

        """
        rtn = pyniNVStrings.n_index(self.m_cptr, sub, start, end, devptr)
        return rtn

    def rindex(self, sub, start=0, end=None, devptr=0):
        """
        Same as rfind but throws an error if arg is not found in all strings.

        Parameters
        ----------
          sub : str
            String to find

          start : int
            Beginning of section to replace.
            Default is beginning of each string.

          end : int
            End of section to replace. Default is end of each string.

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory size must be able to hold at least size() of int32 values.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","world"])

          print(s.rindex('l'))

        Output:

        .. code-block:: python

          [3,3]

        """
        rtn = pyniNVStrings.n_rindex(self.m_cptr, sub, start, end, devptr)
        return rtn

    def find(self, sub, start=0, end=None, devptr=0):
        """
        Find the specified string sub within each string.
        Return -1 for those strings where sub is not found.

        Parameters
        ----------
          sub : str
            String to find

          start : int
            Beginning of section to replace.
            Default is beginning of each string.

          end : int
            End of section to replace. Default is end of each string.

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory size must be able to hold at least size() of int32 values.


        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.find('o'))

        Output:

        .. code-block:: python

          [4,-1,1]

        """
        rtn = pyniNVStrings.n_find(self.m_cptr, sub, start, end, devptr)
        return rtn

    def find_from(self, sub, starts=0, ends=0, devptr=0):
        """
        Find the specified string within each string starting at the
        specified character positions.

        The starts and ends parameters are device memory pointers.
        If specified, each must contain size() of int32 values.

        Returns -1 for those strings where sub is not found.

        Parameters
        ----------
          sub : str
            String to find

          starts : GPU memory pointer
            Pointer to GPU array of int32 values of beginning of sections to
            search, one per string.

          ends : GPU memory pointer
            Pointer to GPU array of int32 values of end of sections to search.
            Use -1 to specify to the end of that string.

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory size must be able to hold at least size() of int32 values.


        Examples
        --------
        .. code-block:: python

          import nvstrings
          import numpy as np
          from numba import cuda

          s = nvstrings.to_device(["hello","there"])
          darr = cuda.to_device(np.asarray([2,3],dtype=np.int32))
          print(s.find_from('e',starts=darr.device_ctypes_pointer.value))

        Output:

        .. code-block:: python

          [-1,4]

        """
        rtn = pyniNVStrings.n_find_from(self.m_cptr, sub, starts, ends, devptr)
        return rtn

    def rfind(self, sub, start=0, end=None, devptr=0):
        """
        Find the specified string within each string.
        Search from the end of the string.

        Return -1 for those strings where sub is not found.

        Parameters
        ----------
          sub : str
            String to find

          start : int
            Beginning of section to replace.
            Default is beginning of each string.

          end : int
            End of section to replace. Default is end of each string.

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.


        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.rfind('o'))

        Output:

        .. code-block:: python

          [4, -1, 1]

        """
        rtn = pyniNVStrings.n_rfind(self.m_cptr, sub, start, end, devptr)
        return rtn

    def findall(self, pat):
        """
        Find all occurrences of regular expression pattern in each string.
        A new array of nvstrings is created for each string in this instance.

        Parameters
        ----------
            pat : str
                The regex pattern used to search for substrings

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hare","bunny","rabbit"])
          for result in s.findall('[ab]'):
            print(result)


        Output:

        .. code-block:: python

          ["a"]
          ["b"]
          ["a","b","b"]

        """
        strs = pyniNVStrings.n_findall(self.m_cptr, pat)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def findall_column(self, pat):
        """
        A new set of nvstrings is created by organizing substring
        results vertically.

        Parameters
        ----------
            pat : str
                The regex pattern to search for substrings

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hare","bunny","rabbit"])
          for result in s.findall_column('[ab]'):
            print(result)


        Output:

        .. code-block:: python

          ["a","b","a"]
          [None,None,"b"]
          [None,None,"b"]

        """
        strs = pyniNVStrings.n_findall_column(self.m_cptr, pat)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def contains(self, pat, regex=True, devptr=0):
        """
        Find the specified string within each string.

        Default expects regex pattern.
        Returns an array of boolean values where
        True if `pat` is found, False if not.

        Parameters
        ----------
          pat : str
            Pattern or string to search for in each string of this instance.

          regex : bool
            If `True`, pat is interpreted as a regex string.
            If `False`, pat is a string to be searched for in each instance.

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Must be able to hold at least size() of np.byte values.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.contains('o'))

        Output:

        .. code-block:: python

          [True, False, True]

        """
        rtn = pyniNVStrings.n_contains(self.m_cptr, pat, regex, devptr)
        return rtn

    def match(self, pat, devptr=0):
        """
        Return array of boolean values where True is set if the specified
        pattern matches the beginning of the corresponding string.

        Parameters
        ----------
          pat : str
            Pattern to find

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory size must be able to hold at least size() of
            np.byte values.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.match('h'))

        Output:

        .. code-block:: python

          [True, False, True]

        """
        rtn = pyniNVStrings.n_match(self.m_cptr, pat, devptr)
        return rtn

    def count(self, pat, devptr=0):
        """
        Count occurrences of pattern in each string.

        Parameters
        ----------
          pat : str
            Pattern to find

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory must be able to hold at least size() of int32 values.

        """
        rtn = pyniNVStrings.n_count(self.m_cptr, pat, devptr)
        return rtn

    def startswith(self, pat, devptr=0):
        """
        Return array of boolean values with True for the strings where the
        specified string is at the beginning.

        Parameters
        ----------
          pat : str
            Pattern to find. Regular expressions are not accepted.

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory must be able to hold at least size() of np.byte values.


        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.startswith('h'))

        Output:

        .. code-block:: python

          [True, False, False]

        """
        rtn = pyniNVStrings.n_startswith(self.m_cptr, pat, devptr)
        return rtn

    def endswith(self, pat, devptr=0):
        """
        Return array of boolean values with True for the strings
        where the specified string is at the end.

        Parameters
        ----------
          pat : str
            Pattern to find. Regular expressions are not accepted.

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory must be able to hold at least size() of np.byte values.


        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.endsswith('d'))

        Output:

        .. code-block:: python

          [False, False, True]

        """
        rtn = pyniNVStrings.n_endswith(self.m_cptr, pat, devptr)
        return rtn

    def extract(self, pat):
        """
        Extract string from the first match of regular expression pat.
        A new array of nvstrings is created for each string in this instance.

        Parameters
        ----------
            pat : str
                The regex pattern with group capture syntax

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["a1","b2","c3"])
          for result in s.extract('([ab])(\\d)'):
            print(result)


        Output:

        .. code-block:: python

          ["a","1"]
          ["b","2"]
          [None,None]

        """
        strs = pyniNVStrings.n_extract(self.m_cptr, pat)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def extract_column(self, pat):
        """
        Extract string from the first match of regular expression pat.
        A new array of nvstrings is created by organizing group results
        vertically.

        Parameters
        ----------
            pat : str
                The regex pattern with group capture syntax

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["a1","b2","c3"])
          for result in s.extract_column('([ab])(\\d)'):
            print(result)


        Output:

        .. code-block:: python

          ["a","b"]
          ["1","2"]
          [None,None]

        """
        strs = pyniNVStrings.n_extract_column(self.m_cptr, pat)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def isalnum(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain
        only alpha-numeric characters.
        Equivalent to: isalpha() or isdigit() or isnumeric() or isdecimal()

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(['1234', 'de', '1.75', '-34', '+9.8', ' '])
          print(s.isalnum())


        Output:

        .. code-block:: python

          [True, True, False, False, False, False]

        """
        rtn = pyniNVStrings.n_isalnum(self.m_cptr, devptr)
        return rtn

    def isalpha(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain
        only alphabetic characters.

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(['1234', 'de', '1.75', '-34', '+9.8', ' '])
          print(s.isalpha())


        Output:

        .. code-block:: python

          [False, True, False, False, False, False]

        """
        rtn = pyniNVStrings.n_isalpha(self.m_cptr, devptr)
        return rtn

    def isdigit(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain
        only decimal and digit characters.

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(['1234', 'de', '1.75', '-34', '+9.8', ' '])
          print(s.isdigit())


        Output:

        .. code-block:: python

          [True, False, False, False, False, False]

        """
        rtn = pyniNVStrings.n_isdigit(self.m_cptr, devptr)
        return rtn

    def isspace(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain
        only whitespace characters.

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(['1234', 'de', '1.75', '-34', '+9.8', ' '])
          print(s.isspace())


        Output:

        .. code-block:: python

          [False, False, False, False, False, True]

        """
        rtn = pyniNVStrings.n_isspace(self.m_cptr, devptr)
        return rtn

    def isdecimal(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain only
        decimal characters -- those that can be used to extract base10 numbers.

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(['1234', 'de', '1.75', '-34', '+9.8', ' '])
          print(s.isdecimal())


        Output:

        .. code-block:: python

          [True, False, False, False, False, False]

        """
        rtn = pyniNVStrings.n_isdecimal(self.m_cptr, devptr)
        return rtn

    def isnumeric(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain
        only numeric characters. These include digit and numeric characters.

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(['1234', 'de', '1.75', '-34', '+9.8', ' '])
          print(s.isnumeric())


        Output:

        .. code-block:: python

          [True, False, False, False, False, False]

        """
        rtn = pyniNVStrings.n_isnumeric(self.m_cptr, devptr)
        return rtn

    def islower(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain
        only lowercase characters.

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(['hello', 'Goodbye'])
          print(s.islower())


        Output:

        .. code-block:: python

          [True, False]

        """
        rtn = pyniNVStrings.n_islower(self.m_cptr, devptr)
        return rtn

    def isupper(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain
        only uppercase characters.

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(['hello', 'Goodbye'])
          print(s.isupper())


        Output:

        .. code-block:: python

          [False, True]

        """
        rtn = pyniNVStrings.n_isupper(self.m_cptr, devptr)
        return rtn

    def translate(self, table):
        """
        Translate individual characters to new characters using
        the provided table.

        Parameters
        ----------
          pat : dict
            Use str.maketrans() to build the mapping table.
            Unspecified characters are unchanged.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","there","world"])
          print(s.translate(str.maketrans('elh','ELH')))

        Output:

        .. code-block:: python

          ['HELLo', 'tHErE', 'worLd]
        """
        rtn = pyniNVStrings.n_translate(self.m_cptr, table)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def sort(self, stype, asc=True):
        """
        Sort this list by name (2) or length (1) or both (3).
        Sorting can help improve performance for other operations.

        Parameters
        ----------
          stype : int
            Type of sort to use.

            If stype is 1, strings will be sorted by length

            If stype is 2, strings will be sorted alphabetically by name

            If stype is 3, strings will be sorted by length and then
            alphabetically

          asc : bool
            Whether to sort ascending (True) or descending (False)

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["aaa", "bb", "aaaabb"])
          print(s.sort(3))

        Output:

        .. code-block:: python

          ['bb', 'aaa', 'aaaabb']

        """
        rtn = pyniNVStrings.n_sort(self.m_cptr, stype, asc)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def order(self, stype, asc=True, devptr=0):
        """
        Sort this list by name (2) or length (1) or both (3).
        This sort only provides the new indexes and does not reorder the
        managed strings.

        Parameters
        ----------
          stype : int
            Type of sort to use.

            If stype is 1, strings will be sorted by length

            If stype is 2, strings will be sorted alphabetically by name

            If stype is 3, strings will be sorted by length and then
            alphabetically

          asc : bool
            Whether to sort ascending (True) or descending (False)

          devptr : GPU memory pointer
                Where index values will be written.
                Must be able to hold at least size() of int32 values.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["aaa", "bb", "aaaabb"])
          print(s.order(2))

        Output:

        .. code-block:: python

          [1, 0, 2]

        """
        rtn = pyniNVStrings.n_order(self.m_cptr, stype, asc, devptr)
        return rtn

    def sublist(self, indexes, count=0):
        """ Calls gather() """
        return self.gather(indexes, count)

    def gather(self, indexes, count=0):
        """
        Return a new list of strings from this instance.

        Parameters
        ----------
          indexes : List of ints or GPU memory pointer
            0-based indexes of strings to return from an nvstrings object

          count : int
            Number of ints if indexes parm is a device pointer.
            Otherwise it is ignored.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.gather([0, 2]))

        Output:

        .. code-block:: python

          ['hello', 'world']

        """
        rtn = pyniNVStrings.n_gather(self.m_cptr, indexes, count)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def remove_strings(self, indexes, count=0):
        """
        Remove the specified strings and return a new instance.

        Parameters
        ----------
          indexes : List of ints
            0-based indexes of strings to remove from an nvstrings object
            If this parameter is pointer to device memory, count parm is
            required.

          count : int
            Number of ints if indexes parm is a device pointer.
            Otherwise it is ignored.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.remove_strings([0, 2]))

        Output:

        .. code-block:: python

          ['there']

        """
        rtn = pyniNVStrings.n_remove_strings(self.m_cptr, indexes, count)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def find_multiple(self, strs, devptr=0):
        """
        Return a 'matrix' of find results for each of the string in the
        strs parameter.

        Each row is an array of integers identifying the first location
        of the corresponding provided string.

        Parameters
        ----------
            strs : nvstrings
                Strings to find in each of the strings in this instance.

            devptr : GPU memory pointer
                Optional device memory pointer to hold the results.

                Memory size must be able to hold at least size()*strs.size()
                of int32 values.

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hare","bunny","rabbit"])
          t = nvstrings.to_device(["a","e","i","o","u"])
          print(s.find_multiple(t))


        Output:

        .. code-block:: python

          [[1, 3, -1, -1, -1], [-1, -1, -1, -1, 1], [1, -1, 4, -1, -1]]

        """
        rtn = pyniNVStrings.n_find_multiple(self.m_cptr, strs, devptr)
        return rtn
