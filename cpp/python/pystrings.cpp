#include <Python.h>
#include <vector>
#include <string>
#include <stdio.h>
#include <exception>
#include <stdexcept>
#include "NVStrings.h"
#include "util.h"

//
// These are C-functions that simply map the python objects to appropriate methods
// in the C++ NVStrings class. There should normally be a 1:1 mapping of
// python nvstrings class methods to C++ NVStrings class method.
// The C-functions here handle marshalling the python object data to/from C/C++
// data structures.
//
// Some cooperation is here around host memory where memory may be freed here
// that is allocated inside the C++ class. This should probably be corrected
// since they may use different allocators and could be risky down the line.
//

// PyArg_VaParse format types are documented here:
// https://docs.python.org/3/c-api/arg.html
bool parse_args( const char* fn, PyObject* pyargs, const char* pyfmt, ... )
{
    va_list args;
    va_start(args,pyfmt);
    bool rtn = (bool)PyArg_VaParse(pyargs,pyfmt,args);
    va_end(args);
    if( !rtn )
        PyErr_Format(PyExc_ValueError,"nvstrings.%s: invalid parameters",fn);
    return rtn;
}

// called by to_device() method in python class
static PyObject* n_createFromHostStrings( PyObject* self, PyObject* args )
{
    PyObject* pystrs = PyTuple_GetItem(args,0); // only one parm expected

    // handle single string
    if( PyObject_TypeCheck(pystrs,&PyUnicode_Type) )
    {
        const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,0));
        return PyLong_FromVoidPtr((void*)NVStrings::create_from_array(&str,1));
    }

    // would be cool if we could check the type is list/array
    //if( !PyObject_TypeCheck(pystrs, &PyArray_Type) )
    //    return PyLong_FromVoidPtr(0); // probably should throw exception

    // handle array of strings
    unsigned int count = (unsigned int)PyList_Size(pystrs);
    const char** list = new const char*[count];
    for( unsigned int idx=0; idx < count; ++idx )
    {
        PyObject* pystr = PyList_GetItem(pystrs,idx);
        if( (pystr == Py_None) || !PyObject_TypeCheck(pystr,&PyUnicode_Type) )
            list[idx] = 0;
        else
            list[idx] = PyUnicode_AsUTF8(pystr);
    }
    //
    //printf("creating %d strings in device memory\n",count);
    NVStrings* thisptr = NVStrings::create_from_array(list,count);
    delete list;
    return PyLong_FromVoidPtr((void*)thisptr);
}

// called by destructor in python class
static PyObject* n_destroyStrings( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings::destroy(tptr);
    return PyLong_FromLong(0);
}

// called in cases where the host code will want the strings back from the device
static PyObject* n_createHostStrings( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    if( count==0 )
        return PyList_New(0);
    std::vector<char*> list(count);
    char** plist = list.data();
    std::vector<int> lens(count);
    size_t totalmem = tptr->byte_count(lens.data(),false);
    std::vector<char> buffer(totalmem+count,0); // null terminates each string
    char* pbuffer = buffer.data();
    size_t offset = 0;
    for( int idx=0; idx < count; ++idx )
    {
        plist[idx] = pbuffer + offset;
        offset += lens[idx]+1; // account for null-terminator; also nulls are -1
    }
    tptr->to_host(plist,0,count);
    PyObject* ret = PyList_New(count);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        char* str = list[idx];
        //printf("[%s]\n",str);
        if( str )
        {
            //printf("{%s}\n",str);
            PyList_SetItem(ret, idx, PyUnicode_FromString((const char*)str));
        }
        else
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
    }
    return ret;
}

static PyObject* n_createFromNVStrings( PyObject* self, PyObject* args )
{
    PyObject* pystrs = PyTuple_GetItem(args,0); // only one parm expected
    if( pystrs == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings: parameter required");
        Py_RETURN_NONE;
    }
    std::vector<NVStrings*> strslist;
    // parameter can be a list of nvstrings instances
    std::string cname = pystrs->ob_type->tp_name;
    if( cname.compare("list")==0 )
    {
        unsigned int count = (unsigned int)PyList_Size(pystrs);
        for( unsigned int idx=0; idx < count; ++idx )
        {
            PyObject* pystr = PyList_GetItem(pystrs,idx);
            cname = pystr->ob_type->tp_name;
            if( cname.compare("nvstrings")!=0 )
            {
                PyErr_Format(PyExc_ValueError,"nvstrings: argument list must contain nvstrings objects");
                Py_RETURN_NONE;
            }
            NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pystr,"m_cptr"));
            if( strs==0 )
            {
                PyErr_Format(PyExc_ValueError,"nvstrings: invalid nvstrings object");
                Py_RETURN_NONE;
            }
            strslist.push_back(strs);
        }
    }
    // or a single nvstrings instance
    else if( cname.compare("nvstrings")==0 )
    {
        NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pystrs,"m_cptr"));
        if( strs==0 )
        {
            PyErr_Format(PyExc_ValueError,"nvstrings: invalid nvstrings object");
            Py_RETURN_NONE;
        }
        strslist.push_back(strs);
    }
    else
    {
        PyErr_Format(PyExc_ValueError,"nvstrings: argument must be nvstrings object");
        Py_RETURN_NONE;
    }

    NVStrings* thisptr = NVStrings::create_from_strings(strslist);
    return PyLong_FromVoidPtr((void*)thisptr);
}

// just for testing and should be removed
static PyObject* n_createFromCSV( PyObject* self, PyObject* args )
{
    std::string csvfile = PyUnicode_AsUTF8(PyTuple_GetItem(args,0));
    unsigned int column = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,1));
    unsigned int lines = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,2));
    unsigned int flags = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,3));
    NVStrings* rtn = createFromCSV(csvfile,column,lines,flags);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// called by from_offsets() method in python class
static PyObject* n_createFromOffsets( PyObject* self, PyObject* args )
{
    PyObject* pysbuf = PyTuple_GetItem(args,0);
    PyObject* pyobuf = PyTuple_GetItem(args,1);
    PyObject* pyscount = PyTuple_GetItem(args,2);
    PyObject* pynbuf = PyTuple_GetItem(args,3);
    PyObject* pyncount = PyTuple_GetItem(args,4);

    //
    if( (pysbuf == Py_None) || (pyobuf == Py_None) )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings: missing parameter");
        Py_RETURN_NONE;
    }

    const char* sbuffer = 0;
    const int* obuffer = 0;
    const unsigned char* nbuffer = 0;
    int scount = (int)PyLong_AsLong(pyscount);
    int ncount = 0;

    Py_buffer sbuf, obuf, nbuf;
    if( PyObject_CheckBuffer(pysbuf) )
    {
        PyObject_GetBuffer(pysbuf,&sbuf,PyBUF_SIMPLE);
        sbuffer = (const char*)sbuf.buf;
    }
    else
        sbuffer = (const char*)PyLong_AsVoidPtr(pysbuf);

    if( PyObject_CheckBuffer(pyobuf) )
    {
        PyObject_GetBuffer(pyobuf,&obuf,PyBUF_SIMPLE);
        obuffer = (const int*)obuf.buf;
    }
    else
        obuffer = (const int*)PyLong_AsVoidPtr(pyobuf);

    if( PyObject_CheckBuffer(pynbuf) )
    {
        PyObject_GetBuffer(pynbuf,&nbuf,PyBUF_SIMPLE);
        nbuffer = (const unsigned char*)nbuf.buf;
    }
    else if( pynbuf != Py_None )
    {
        nbuffer = (const unsigned char*)PyLong_AsVoidPtr(pynbuf);
        ncount = (int)PyLong_AsLong(pyncount);
    }

    //printf(" ptrs=%p,%p,%p\n",sbuffer,obuffer,nbuffer);
    //printf(" scount=%d,ncount=%d\n",scount,ncount);
    // create strings object from these buffers
    NVStrings* rtn = NVStrings::create_from_offsets(sbuffer,scount,obuffer,nbuffer,ncount);

    if( PyObject_CheckBuffer(pysbuf) )
        PyBuffer_Release(&sbuf);
    if( PyObject_CheckBuffer(pyobuf) )
        PyBuffer_Release(&obuf);
    if( PyObject_CheckBuffer(pynbuf) )
        PyBuffer_Release(&nbuf);

    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_createFromIntegers( PyObject* self, PyObject* args )
{
    PyObject* pyvals = PyTuple_GetItem(args,0);
    PyObject* pycount = PyTuple_GetItem(args,1);
    PyObject* pybmem = PyTuple_GetItem(args,2);

    bool bdevmem = (bool)PyObject_IsTrue(pybmem);
    NVStrings* rtn = 0;
    std::string cname = pyvals->ob_type->tp_name;
    if( cname.compare("list")==0 )
    {
        unsigned int elems = (unsigned int)PyList_Size(pyvals);
        std::vector<int> values;
        for( unsigned int idx=0; idx < elems; ++idx )
        {
            PyObject* pyidx = PyList_GetItem(pyvals,idx);
            values.push_back((int)PyLong_AsLong(pyidx));
        }
        //
        rtn = NVStrings::itos(values.data(),elems,false);
    }
    else if( cname.compare("DeviceNDArray")==0 )
    {
        PyObject* pysize = PyObject_GetAttr(pyvals,PyUnicode_FromString("alloc_size"));
        PyObject* pydcp = PyObject_GetAttr(pyvals,PyUnicode_FromString("device_ctypes_pointer"));
        PyObject* pyptr = PyObject_GetAttr(pydcp,PyUnicode_FromString("value"));
        unsigned int count = (unsigned int)(PyLong_AsLong(pysize)/sizeof(int));
        int* values = 0;
        if( pyptr != Py_None )
            values = (int*)PyLong_AsVoidPtr(pyptr);
        rtn = NVStrings::itos(values,count);
    }
    else if( PyObject_CheckBuffer(pyvals) )
    {
        Py_buffer mbuf;
        PyObject_GetBuffer(pyvals,&mbuf,PyBUF_SIMPLE);
        int* values = (int*)mbuf.buf;
        unsigned int count = (unsigned int)(mbuf.len/sizeof(int));
        rtn = NVStrings::itos(values,count,bdevmem);
        PyBuffer_Release(&mbuf);
    }
    else if( cname.compare("int")==0 ) // device pointer directly
    {                                  // for consistency with other methods
        int* values = (int*)PyLong_AsVoidPtr(pyvals);
        unsigned int count = (unsigned int)PyLong_AsLong(pycount);
        rtn = NVStrings::itos(values,count,bdevmem);
    }
    else
    {
        //printf("%s\n",cname.c_str());
        PyErr_Format(PyExc_TypeError,"nvstrings: unknown type %s",cname.c_str());
    }
    //
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// called by from_offsets() method in python class
static PyObject* n_create_offsets( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pysbuf = PyTuple_GetItem(args,1);
    PyObject* pyobuf = PyTuple_GetItem(args,2);
    PyObject* pynbuf = PyTuple_GetItem(args,3);

    //
    if( (pysbuf == Py_None) || (pyobuf == Py_None) )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings: missing parameter");
        Py_RETURN_NONE;
    }

    char* sbuffer = 0;
    int* obuffer = 0;
    unsigned char* nbuffer = 0;
    int ncount = 0;

    Py_buffer sbuf, obuf, nbuf;
    if( PyObject_CheckBuffer(pysbuf) )
    {
        PyObject_GetBuffer(pysbuf,&sbuf,PyBUF_SIMPLE);
        sbuffer = (char*)sbuf.buf;
    }
    else
        sbuffer = (char*)PyLong_AsVoidPtr(pysbuf);

    if( PyObject_CheckBuffer(pyobuf) )
    {
        PyObject_GetBuffer(pyobuf,&obuf,PyBUF_SIMPLE);
        obuffer = (int*)obuf.buf;
    }
    else
        obuffer = (int*)PyLong_AsVoidPtr(pyobuf);

    if( PyObject_CheckBuffer(pynbuf) )
    {
        PyObject_GetBuffer(pynbuf,&nbuf,PyBUF_SIMPLE);
        nbuffer = (unsigned char*)nbuf.buf;
    }
    else if( pynbuf != Py_None )
        nbuffer = (unsigned char*)PyLong_AsVoidPtr(pynbuf);

    PyObject* pybmem = PyTuple_GetItem(args,4);
    bool bdevmem = (bool)PyObject_IsTrue(pybmem);

    //printf(" ptrs=%p,%p,%p\n",sbuffer,obuffer,nbuffer);
    //printf(" scount=%d,ncount=%d\n",scount,ncount);
    // create strings object from these buffers
    tptr->create_offsets(sbuffer,obuffer,nbuffer,bdevmem);

    if( PyObject_CheckBuffer(pysbuf) )
        PyBuffer_Release(&sbuf);
    if( PyObject_CheckBuffer(pyobuf) )
        PyBuffer_Release(&obuf);
    if( PyObject_CheckBuffer(pynbuf) )
        PyBuffer_Release(&nbuf);

    Py_RETURN_NONE;
}

static PyObject* n_size( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    return PyLong_FromLong(count);
}

// return the length of each string
static PyObject* n_len( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        tptr->len(devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }

    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    int* rtn = new int[count];
    tptr->len(rtn,false);
    for(unsigned int idx=0; idx < count; idx++)
    {
        int val = rtn[idx];
        if( val < 0 )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
        else
            PyList_SetItem(ret, idx, PyLong_FromLong((long)val));
    }
    delete rtn;
    return ret;
}

static PyObject* n_byte_count( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    int* memptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    bool bdevmem = (bool)PyObject_IsTrue(PyTuple_GetItem(args,2));

    size_t rtn = tptr->byte_count(memptr,bdevmem);
    return PyLong_FromLong((long)rtn);
}

// return the number of nulls
static PyObject* n_null_count( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool ben = (bool)PyObject_IsTrue(PyTuple_GetItem(args,1));
    unsigned int nulls = tptr->get_nulls(0,ben,false);
    return PyLong_FromLong((long)nulls);
}

// compare a string to the list of strings
// a future method could compare a list to another list
static PyObject* n_compare( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    //
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    if( devptr )
    {
        tptr->compare(str,devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }
    //
    int* rtn = new int[count];
    tptr->compare(str,rtn,false);
    for(size_t idx=0; idx < count; idx++)
    {
        int val = rtn[idx];
        //PyList_SetItem(ret, idx, Py_None);
        PyList_SetItem(ret, idx, PyLong_FromLong((long)val));
    }
    delete rtn;
    return ret;
}

//
static PyObject* n_hash( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    unsigned int* devptr = (unsigned int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        tptr->hash(devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }

    // copy to host option
    unsigned int* rtn = new unsigned int[count];
    tptr->hash(rtn,false);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyLong_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

// convert the strings to integers
static PyObject* n_stoi( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    //
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        tptr->stoi(devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }

    // copy to host option
    int* rtn = new int[count];
    tptr->stoi(rtn,false);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyLong_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

// convert the strings to floats
static PyObject* n_stof( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    //
    float* devptr = (float*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        tptr->stof(devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }
    float* rtn = new float[count];
    tptr->stof(rtn,false);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyFloat_FromDouble((double)rtn[idx]));
    delete rtn;
    return ret;
}

// convert the strings with hex characters to integers
static PyObject* n_htoi( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    //
    unsigned int* devptr = (unsigned int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        tptr->htoi(devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }

    // copy to host option
    unsigned int* rtn = new unsigned int[count];
    tptr->htoi(rtn,false);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyLong_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

// concatenate the given string to the end of all the strings
static PyObject* n_cat( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* argOthers = PyTuple_GetItem(args,1);
    PyObject* argSep = PyTuple_GetItem(args,2);
    PyObject* argNaRep = PyTuple_GetItem(args,3);
    //PyObject* argJoin = PyTuple_GetItem(args,4);

    const char* sep = "";
    if( argSep != Py_None )
        sep = PyUnicode_AsUTF8(argSep);

    const char* narep = 0;
    if( argNaRep != Py_None )
        narep = PyUnicode_AsUTF8(argNaRep);

    if( argOthers == Py_None )
    {
        // this is just a join -- need to account for the other parms too
        NVStrings* rtn = tptr->join(sep,narep);
        if( rtn )
            return PyLong_FromVoidPtr((void*)rtn);
        Py_RETURN_NONE;
    }

    //printf("HasAttrString(m_cptr)=%d\n", PyObject_HasAttrString(argOthers,"m_cptr"));
    NVStrings* others = 0;
    //printf("arg.ob_type.tp_name=[%s]\n", argOthers->ob_type->tp_name);
    std::string cname = argOthers->ob_type->tp_name;
    if( cname.compare("list")==0 )
    {
        unsigned int count = (unsigned int)PyList_Size(argOthers);
        if( count==0 )
        {
            PyErr_Format(PyExc_ValueError,"nvstrings.cat empty argument list");
            Py_RETURN_NONE;
        }

        if( count != (int)tptr->size() )
        {
            PyErr_Format(PyExc_ValueError,"nvstrings.cat list size must match");
            Py_RETURN_NONE;
        }

        const char** list = new const char*[count];
        for( unsigned int idx=0; idx < count; ++idx )
        {
            PyObject* pystr = PyList_GetItem(argOthers,idx);
            if( (pystr == Py_None) || !PyObject_TypeCheck(pystr,&PyUnicode_Type) )
                list[idx] = 0;
            else
                list[idx] = PyUnicode_AsUTF8(pystr);
        }
        others = NVStrings::create_from_array(list,count);
        delete list;
    }
    //
    if( cname.compare("nvstrings")==0 )
    {
        others = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(argOthers,"m_cptr"));
        if( !others )
        {
            PyErr_Format(PyExc_ValueError,"nvstrings.cat invalid parameter");
            Py_RETURN_NONE;
        }
        //printf("others count=%d\n",others->size());
        if( others->size() != tptr->size() )
        {
            PyErr_Format(PyExc_ValueError,"nvstrings.cat list size must match");
            Py_RETURN_NONE;
        }
    }

    //
    NVStrings* rtn = 0;
    if( others )
    {
        rtn = tptr->cat(others,sep,narep);
        if( cname.compare("list")==0 )
            NVStrings::destroy(others); // destroy it if we made it (above)
    }

    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// split each string into newer strings
// this will return an array of NVStrings to be wrapped in nvstrings
static PyObject* n_split( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* delimiter = " ";
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        delimiter = PyUnicode_AsUTF8(argOpt);
    int maxsplit = -1;
    argOpt = PyTuple_GetItem(args,2);
    if( argOpt != Py_None )
        maxsplit = (int)PyLong_AsLong(argOpt);

    std::vector<NVStrings*> results;
    tptr->split(delimiter,maxsplit,results);
    //
    PyObject* ret = PyList_New(tptr->size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

// another split but from the right
static PyObject* n_rsplit( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* delimiter = " ";
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        delimiter = PyUnicode_AsUTF8(argOpt);
    int maxsplit = -1;
    argOpt = PyTuple_GetItem(args,2);
    if( argOpt != Py_None )
        maxsplit = (int)PyLong_AsLong(argOpt);

    std::vector<NVStrings*> results;
    tptr->rsplit(delimiter,maxsplit,results);
    //
    PyObject* ret = PyList_New(tptr->size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret, idx, PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

//
static PyObject* n_partition( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* delimiter = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    //
    std::vector<NVStrings*> results;
    tptr->partition(delimiter,results);
    //
    PyObject* ret = PyList_New(tptr->size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

//
static PyObject* n_rpartition( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* delimiter = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    //
    std::vector<NVStrings*> results;
    tptr->rpartition(delimiter,results);
    //
    PyObject* ret = PyList_New(tptr->size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

static PyObject* n_split_column( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* delimiter = " ";
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        delimiter = PyUnicode_AsUTF8(argOpt);
    int maxsplit = -1;
    argOpt = PyTuple_GetItem(args,2);
    if( argOpt != Py_None )
        maxsplit = (int)PyLong_AsLong(argOpt);

    std::vector<NVStrings*> results;
    int columns = (int)tptr->split_column(delimiter,maxsplit,results);
    //
    PyObject* ret = PyList_New(columns);
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

static PyObject* n_rsplit_column( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* delimiter = " ";
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        delimiter = PyUnicode_AsUTF8(argOpt);
    int maxsplit = -1;
    argOpt = PyTuple_GetItem(args,2);
    if( argOpt != Py_None )
        maxsplit = (int)PyLong_AsLong(argOpt);

    std::vector<NVStrings*> results;
    int columns = (int)tptr->rsplit_column(delimiter,maxsplit,results);
    //
    PyObject* ret = PyList_New(columns);
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

// return a single character
// this will return a new NVStrings array where the strings are all single characters
static PyObject* n_get( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int position = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,1));
    NVStrings* rtn = tptr->get(position);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// repeat each string a number of times
static PyObject* n_repeat( PyObject* self, PyObject* args )
{
    PyObject* vo = 0;
    unsigned int count = 0;
    if( !parse_args("repeat",args,"OI",&vo,&count) )
        Py_RETURN_NONE;
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(vo);
    NVStrings* rtn = tptr->repeat(count);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// add padding around strings to a fixed size
static PyObject* n_pad( PyObject* self, PyObject* args )
{
    PyObject* vo = 0;
    unsigned int width = 0;
    const char* side = 0;
    const char* fillchar = 0;
    if( !parse_args("pad",args,"OIzz",&vo,&width,&side,&fillchar) )
        Py_RETURN_NONE;
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(vo);
    NVStrings::padside ps = NVStrings::left;
    std::string sside = side;
    if( sside.compare("right")==0 )
        ps = NVStrings::right;
    else if( sside.compare("both")==0 )
        ps = NVStrings::both;
    NVStrings* rtn = tptr->pad(width,ps,fillchar);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// left-justify (and right pad) each string
static PyObject* n_ljust( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int width = 0;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        width = (unsigned int)PyLong_AsLong(argOpt);
    NVStrings* rtn = tptr->ljust(width);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// center each string and pad right/left
static PyObject* n_center( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int width = 0;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        width = (unsigned int)PyLong_AsLong(argOpt);
    NVStrings* rtn = tptr->center(width);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// right justify each string (and left pad)
static PyObject* n_rjust( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int width = 0;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        width = (unsigned int)PyLong_AsLong(argOpt);
    NVStrings* rtn = tptr->rjust(width);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// zero pads strings correctly that contain numbers
static PyObject* n_zfill( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int width = 0;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        width = (unsigned int)PyLong_AsLong(argOpt);
    NVStrings* rtn = tptr->zfill(width);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// this attempts to do some kind of line wrapping by inserting new line characters
static PyObject* n_wrap( PyObject* self, PyObject* args )
{
    unsigned int width = 0;
    PyObject* vo = 0;
    if( !parse_args("wrap",args,"OI",&vo,&width) )
        Py_RETURN_NONE;
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(vo);
    NVStrings* rtn = tptr->wrap(width);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// returns substring of each string
static PyObject* n_slice( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    int start = PyLong_AsLong(PyTuple_GetItem(args,1));
    int end = -1, step = 1;
    PyObject* argOpt = PyTuple_GetItem(args,2);
    if( argOpt != Py_None )
        end = (int)PyLong_AsLong(argOpt);
    argOpt = PyTuple_GetItem(args,3);
    if( argOpt != Py_None )
        step = (int)PyLong_AsLong(argOpt);
    NVStrings* rtn = tptr->slice(start,end,step);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// returns substring of each string using individual position values
static PyObject* n_slice_from( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    int* starts = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    int* ends = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    NVStrings* rtn = tptr->slice_from(starts,ends);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// replaces the given range with the given string
static PyObject* n_slice_replace( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    int start = 0, end = -1;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        start = (int)PyLong_AsLong(argOpt);
    argOpt = PyTuple_GetItem(args,2);
    if( argOpt != Py_None )
        end = (int)PyLong_AsLong(argOpt);
    const char* repl = 0;
    argOpt = PyTuple_GetItem(args,3);
    if( argOpt != Py_None )
        repl = PyUnicode_AsUTF8(argOpt);
    //
    NVStrings* rtn = tptr->slice_replace(repl,start,end);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// replace the string specified (if found) with the target string
static PyObject* n_replace( PyObject* self, PyObject* args )
{
    PyObject* vo = 0;      // self pointer   = O
    const char* pat = 0;   // cannot be null = s
    const char* repl = 0;  // can be null    = z
    int maxrepl = -1;      // integer        = i
    int bregex = true;     // boolean        = p (do not use bool type here)
    if( !parse_args("replace",args,"Oszip",&vo,&pat,&repl,&maxrepl,&bregex) )
        Py_RETURN_NONE;
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(vo);
    NVStrings* rtn = 0;
    if( bregex )
    {
        rtn = tptr->replace_re(pat,repl,(int)maxrepl);
        if( rtn==0 )
            PyErr_Format(PyExc_ValueError,"nvstrings.replace regex pattern is too long");
    }
    else
        rtn = tptr->replace(pat,repl,(int)maxrepl);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

//
static PyObject* n_replace_with_backrefs( PyObject* self, PyObject* args )
{
    PyObject* vo = 0;      // self pointer   = O
    const char* pat = 0;   // cannot be null = s
    const char* repl = 0;  // can be null    = z
    if( !parse_args("replace",args,"Osz",&vo,&pat,&repl) )
        Py_RETURN_NONE;
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(vo);
    NVStrings* rtn = 0;
    rtn = tptr->replace_with_backrefs(pat,repl);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// strip specific characters from the beginning of each string
static PyObject* n_lstrip( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* to_strip = 0;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        to_strip = PyUnicode_AsUTF8(argOpt);
    NVStrings* rtn = tptr->lstrip(to_strip);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// strip characters from the beginning and the end of each string
static PyObject* n_strip( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* to_strip = 0;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        to_strip = PyUnicode_AsUTF8(argOpt);
    NVStrings* rtn = tptr->strip(to_strip);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// right strip characters from the end of each string
static PyObject* n_rstrip( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* to_strip = 0;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        to_strip = PyUnicode_AsUTF8(argOpt);
    NVStrings* rtn = tptr->rstrip(to_strip);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// lowercase each string in place
static PyObject* n_lower( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings* rtn = tptr->lower();
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// uppercase each string in place
static PyObject* n_upper( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings* rtn = tptr->upper();
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// capitalize the first character of each string
static PyObject* n_capitalize( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings* rtn = tptr->capitalize();
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// swap the upper/lower case of each string's characters
static PyObject* n_swapcase( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings* rtn = tptr->swapcase();
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// title-case each string
static PyObject* n_title( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings* rtn = tptr->title();
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// search for the given string and return the positions it was found in each string
// https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.find.html
static PyObject* n_find( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    int start = (int)PyLong_AsLong(PyTuple_GetItem(args,2));
    int end = -1;
    PyObject* argEnd = PyTuple_GetItem(args,3);
    if( argEnd != Py_None )
        end = (int)PyLong_AsLong(argEnd);
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,4));
    if( devptr )
    {
        tptr->find(str,start,end,devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    int* rtn = new int[count];
    tptr->find(str,start,end,rtn,false);
    for(unsigned int idx=0; idx < count; idx++)
    {
        int val = rtn[idx];
        if( val < -1 )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
        else
            PyList_SetItem(ret, idx, PyLong_FromLong((long)val));
    }
    delete rtn;
    return ret;
}

// this was created out of a need to search for the 2nd occurrence of a string
static PyObject* n_find_from( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    int* starts = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    int* ends = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,3));

    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,4));
    if( devptr )
    {
        tptr->find_from(str,starts,ends,devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    int* rtn = new int[count];
    tptr->find_from(str,starts,ends,rtn,false);
    for(unsigned int idx=0; idx < count; idx++)
    {
        int val = rtn[idx];
        if( val < -1 )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
        else
            PyList_SetItem(ret, idx, PyLong_FromLong((long)val));
    }
    delete rtn;
    return ret;
}

// right-search for the given string and return the positions it was found in each string
static PyObject* n_rfind( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    int start = (int)PyLong_AsLong(PyTuple_GetItem(args,2));
    int end = -1;
    PyObject* argEnd = PyTuple_GetItem(args,3);
    if( argEnd != Py_None )
        end = (int)PyLong_AsLong(argEnd);
    //
    unsigned int count = tptr->size();
    //
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,4));
    if( devptr )
    {
        tptr->rfind(str,start,end,devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    int* rtn = new int[count];
    tptr->rfind(str,start,end,rtn,false);
    for(unsigned int idx=0; idx < count; idx++)
    {
        int val = rtn[idx];
        if( val < -1 )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
        else
            PyList_SetItem(ret, idx, PyLong_FromLong((long)val));
    }
    delete rtn;
    return ret;
}

// return position of each string provided
static PyObject* n_find_multiple( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* argStrs = PyTuple_GetItem(args,1);
    if( argStrs == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.find_multiple strs argument must be specified");
        Py_RETURN_NONE;
    }
    NVStrings* strs = 0;
    std::string cname = argStrs->ob_type->tp_name;
    if( cname.compare("nvstrings")==0 )
        strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(argStrs,"m_cptr"));
    else if( cname.compare("list")==0 )
    {
        unsigned int count = (unsigned int)PyList_Size(argStrs);
        if( count )
        {
            const char** list = new const char*[count];
            for( unsigned int idx=0; idx < count; ++idx )
            {
                PyObject* pystr = PyList_GetItem(argStrs,idx);
                if( (pystr == Py_None) || !PyObject_TypeCheck(pystr,&PyUnicode_Type) )
                    list[idx] = 0;
                else
                    list[idx] = PyUnicode_AsUTF8(pystr);
            }
            strs = NVStrings::create_from_array(list,count);
            delete list;
        }
    }
    //
    if( !strs )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.find_multiple invalid strs parameter");
        Py_RETURN_NONE;
    }
    if( strs->size()==0 )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.find_multiple empty strs list");
        Py_RETURN_NONE;
    }

    // resolve output pointer
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    if( devptr )
    {
        tptr->find_multiple(*strs,devptr);
        if( cname.compare("list")==0 )
            NVStrings::destroy(strs); // destroy it if we made it (above)
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int rows = tptr->size();
    PyObject* ret = PyList_New(rows);
    if( rows==0 )
    {
        if( cname.compare("list")==0 )
            NVStrings::destroy(strs); // destroy it if we made it (above)
        return ret;
    }
    //
    unsigned int columns = strs->size();
    int* rtn = new int[rows*columns];
    tptr->find_multiple(*strs,rtn,false);
    for(unsigned int idx=0; idx < rows; ++idx)
    {
        PyObject* row = PyList_New(columns);
        for( unsigned int jdx=0; jdx < columns; ++jdx )
        {
            int val = rtn[(idx*columns)+jdx];
            if( val < -1 )
            {
                Py_INCREF(Py_None);
                PyList_SetItem(row, jdx, Py_None);
            }
            else
                PyList_SetItem(row, jdx, PyLong_FromLong((long)val));
        }
        PyList_SetItem(ret, idx, row);
    }
    delete rtn;
    //
    if( cname.compare("list")==0 )
        NVStrings::destroy(strs); // destroy it if we made it (above)
    return ret;
}

// this is the same as find but throws an error if string is not found
// https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.index.html
static PyObject* n_index( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    int start = (int)PyLong_AsLong(PyTuple_GetItem(args,2));
    int end = -1;
    PyObject* argEnd = PyTuple_GetItem(args,3);
    if( argEnd != Py_None )
        end = (int)PyLong_AsLong(argEnd);
    //
    unsigned int count = tptr->size();
    //
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,4));
    if( devptr )
    {
        unsigned int success = tptr->find(str,start,end,devptr);
        if( success != count )
            PyErr_Format(PyExc_ValueError,"nvstrings.index: [%s] not found in %d elements",str,(int)(count-success));
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    int* rtn = new int[count];
    tptr->find(str,start,end,rtn,false);
    for(unsigned int idx=0; idx < count; idx++)
    {
        int val = rtn[idx];
        if( val < -1 )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
        else if( val >= 0 )
            PyList_SetItem(ret, idx, PyLong_FromLong((long)val));
        else
        {
            PyErr_Format(PyExc_ValueError,"nvstrings.index: [%s] not found in element %d",str,(int)idx);
            break;
        }
    }
    delete rtn;
    return ret;
}

// same as rfind excepts throws an error if string is not found
static PyObject* n_rindex( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    int start = (int)PyLong_AsLong(PyTuple_GetItem(args,2));
    int end = -1;
    PyObject* argEnd = PyTuple_GetItem(args,3);
    if( argEnd != Py_None )
        end = (int)PyLong_AsLong(argEnd);
    //
    unsigned int count = tptr->size();
    //
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,4));
    if( devptr )
    {
        size_t success = tptr->rfind(str,start,end,devptr);
        if( success != count )
            PyErr_Format(PyExc_ValueError,"nvstrings.rindex: [%s] not found in %d elements",str,(int)(count-success));
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    int* rtn = new int[count];
    tptr->rfind(str,start,end,rtn,false);
    for(unsigned int idx=0; idx < count; idx++)
    {
        int val = rtn[idx];
        if( val < -1 )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
        else if( val >= 0 )
            PyList_SetItem(ret, idx, PyLong_FromLong((long)val));
        else
        {
            PyErr_Format(PyExc_ValueError,"nvstrings.rindex: [%s] not found in element %d",str,(int)idx);
            break;
        }
    }
    delete rtn;
    return ret;
}

// this will return an array of NVStrings to be wrapped in nvstrings
static PyObject* n_findall( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* pat = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));

    std::vector<NVStrings*> results;
    int rc = tptr->findall(pat,results);
    if( rc==-2 )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.findall regex pattern is too long");
        Py_RETURN_NONE;
    }
    //
    PyObject* ret = PyList_New(results.size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

// same but column-major groupings of results
static PyObject* n_findall_column( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* pat = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));

    std::vector<NVStrings*> results;
    int rc = tptr->findall_column(pat,results);
    if( rc==-2 )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.findall_column regex pattern is too long");
        Py_RETURN_NONE;
    }
    //
    PyObject* ret = PyList_New(results.size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}


// This can take a regex string too
// https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.contains.html
static PyObject* n_contains( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    //
    bool bregex = (bool)PyObject_IsTrue(PyTuple_GetItem(args,2));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,3));
    int rc = 0;
    if( devptr )
    {
        if( bregex )
        {
            rc = tptr->contains_re(str,devptr);
            if( rc==-2 )
                PyErr_Format(PyExc_ValueError,"nvstrings.contains regex pattern is too long");
        }
        else
            rc = tptr->contains(str,devptr);
        if( rc < 0 )
            Py_RETURN_NONE;
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    if( count==0 )
        return PyList_New(0);
    bool* rtn = new bool[count];
    if( bregex )
    {
        rc = tptr->contains_re(str,rtn,false);
        if( rc==-2 )
            PyErr_Format(PyExc_ValueError,"nvstrings.contains regex pattern is too long");
    }
    else
        rc = tptr->contains(str,rtn,false);
    if( rc < 0 )
    {
        delete rtn;
        Py_RETURN_NONE;
    }
    PyObject* ret = PyList_New(count);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

static PyObject* n_match( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    //
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    int rc = 0;
    if( devptr )
    {
        rc = tptr->match(str,devptr);
        if( rc==-2 )
            PyErr_Format(PyExc_ValueError,"nvstrings.match regex pattern is too long");
        if( rc < 0 )
            Py_RETURN_NONE;
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    if( count==0 )
        return PyList_New(0);
    bool* rtn = new bool[count];
    rc = tptr->match(str,rtn,false);
    if( rc==-2 )
        PyErr_Format(PyExc_ValueError,"nvstrings.match regex pattern is too long");
    if( rc < 0 )
    {
        delete rtn;
        Py_RETURN_NONE;
    }
    PyObject* ret = PyList_New(count);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

//
static PyObject* n_startswith( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    if( devptr )
    {
        tptr->startswith(str,devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    tptr->startswith(str,rtn,false);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

//
static PyObject* n_endswith( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    if( devptr )
    {
        tptr->endswith(str,devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    tptr->endswith(str,rtn,false);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

//
static PyObject* n_count( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    if( devptr )
    {
        int rc = tptr->count_re(str,devptr);
        if( rc==-2 )
            PyErr_Format(PyExc_ValueError,"nvstrings.count regex pattern is too long");
        if( rc < 0 )
            Py_RETURN_NONE;
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    if( count==0 )
        return PyList_New(0);
    int* rtn = new int[count];
    int rc = tptr->count_re(str,rtn,false);
    if( rc==-2 )
        PyErr_Format(PyExc_ValueError,"nvstrings.count regex pattern is too long");
    if( rc < 0 )
    {
        delete rtn;
        Py_RETURN_NONE;
    }
    //
    PyObject* ret = PyList_New(count);
    for(size_t idx=0; idx < count; idx++)
    {
        int val = rtn[idx];
        if( val >= 0 )
            PyList_SetItem(ret, idx, PyLong_FromLong((long)val));
        else
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
    }
    delete rtn;
    return ret;
}

// this will return an array of NVStrings to be wrapped in nvstrings
static PyObject* n_extract( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* pat = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));

    std::vector<NVStrings*> results;
    int rc = tptr->extract(pat,results);
    if( rc==-2 )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.extract regex pattern is too long");
        Py_RETURN_NONE;
    }
    //
    PyObject* ret = PyList_New(results.size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

// same but column-major groupings of results
static PyObject* n_extract_column( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* pat = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));

    std::vector<NVStrings*> results;
    int rc = tptr->extract_column(pat,results);
    if( rc==-2 )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.extract_column regex pattern is too long");
        Py_RETURN_NONE;
    }
    //
    PyObject* ret = PyList_New(results.size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

// This translates each character based on a given table.
// The table can be an array of pairs--array with 2 values or a dictionary created by str.maketrans()
static PyObject* n_translate( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pytable = PyTuple_GetItem(args,1);
    std::string cname = pytable->ob_type->tp_name; // list or dict

    unsigned int count = 0;
    std::vector< std::pair<unsigned,unsigned> > table;
    if( cname.compare("list")==0 )
    {
        // convert table parm into pair<unsigned,unsigned> array
        count = (unsigned int)PyList_Size(pytable);
        for( unsigned int idx=0; idx < count; ++idx )
        {
            PyObject* pyentry = PyList_GetItem(pytable,idx);
            if( PyList_Size(pyentry)!=2 )
            {
                PyErr_Format(PyExc_ValueError,"nvstrings.translate: invalid map entry");
                Py_RETURN_NONE;
            }
            std::pair<unsigned,unsigned> entry;
            PyUnicode_AsWideChar(PyList_GetItem(pyentry,0),(wchar_t*)&(entry.first),1);
            PyObject* pysecond = PyList_GetItem(pyentry,1);
            if( pysecond != Py_None )
                PyUnicode_AsWideChar(pysecond,(wchar_t*)&(entry.second),1);
            else
                entry.second = 0;
            table.push_back(entry);
        }
    }
    else if( cname.compare("dict")==0 )
    {
        count = (unsigned int)PyDict_Size(pytable);
        PyObject* items = PyDict_Items(pytable);
        for( unsigned int idx=0; idx < count; ++idx )
        {
            PyObject* item = PyList_GetItem(items,idx);
            std::pair<unsigned,unsigned> entry;
            entry.first = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(item,0));
            PyObject* item1 = PyTuple_GetItem(item,1);
            if( item1 != Py_None )
                entry.second = (unsigned)PyLong_AsUnsignedLong(item1);
            else
                entry.second = 0;
            table.push_back(entry);
        }
    }
    else
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.translate: invalid argument type");
        Py_RETURN_NONE;
    }
    //
    NVStrings* rtn = tptr->translate(table.data(),count);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// joins two strings with these strings in the middle
static PyObject* n_join( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* delim = "";
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        delim = PyUnicode_AsUTF8(argOpt);
    NVStrings* rtn = tptr->join(delim);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// sorts the strings by length/name
static PyObject* n_sort( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings::sorttype stype = (NVStrings::sorttype)PyLong_AsLong(PyTuple_GetItem(args,1));
    int asc = (int)PyLong_AsLong(PyTuple_GetItem(args,2));
    NVStrings* rtn = tptr->sort(stype,(bool)asc);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// like sort but returns new index order only
static PyObject* n_order( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings::sorttype stype = (NVStrings::sorttype)PyLong_AsLong(PyTuple_GetItem(args,1));
    int asc = (int)PyLong_AsLong(PyTuple_GetItem(args,2));
    unsigned int* devptr = (unsigned int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,3));
    if( devptr )
    {
        tptr->order(stype,(bool)asc,devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }

    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    unsigned int* rtn = new unsigned int[count];
    tptr->order(stype,(bool)asc,rtn,false);
    for(unsigned int idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyLong_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

//
static PyObject* n_gather( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pyidxs = PyTuple_GetItem(args,1);
    std::string cname = pyidxs->ob_type->tp_name;
    NVStrings* rtn = 0;
    try
    {
        if( cname.compare("list")==0 )
        {
            unsigned int count = (unsigned int)PyList_Size(pyidxs);
            int* indexes = new int[count];
            for( unsigned int idx=0; idx < count; ++idx )
            {
                PyObject* pyidx = PyList_GetItem(pyidxs,idx);
                indexes[idx] = (int)PyLong_AsLong(pyidx);
            }
            //
            rtn = tptr->gather(indexes,count,false);
            delete indexes;
        }
        else if( cname.compare("DeviceNDArray")==0 )
        {
            PyObject* pysize = PyObject_GetAttr(pyidxs,PyUnicode_FromString("alloc_size"));
            PyObject* pydcp = PyObject_GetAttr(pyidxs,PyUnicode_FromString("device_ctypes_pointer"));
            PyObject* pyptr = PyObject_GetAttr(pydcp,PyUnicode_FromString("value"));
            unsigned int count = (unsigned int)(PyLong_AsLong(pysize)/sizeof(int));
            int* indexes = 0;
            if( pyptr != Py_None )
                indexes = (int*)PyLong_AsVoidPtr(pyptr);
            //printf("device-array: %p,%u\n",indexes,count);
            rtn = tptr->gather(indexes,count);
        }
        else if( PyObject_CheckBuffer(pyidxs) )
        {
            Py_buffer pybuf;
            PyObject_GetBuffer(pyidxs,&pybuf,PyBUF_SIMPLE);
            int* indexes = (int*)pybuf.buf;
            unsigned int count = (unsigned int)(pybuf.len/sizeof(int));
            //printf("buffer: %p,%u\n",indexes,count);
            rtn = tptr->gather(indexes,count,false);
            PyBuffer_Release(&pybuf);
        }
        else if( cname.compare("int")==0 ) // device pointer directly
        {                                  // for consistency with other methods
            int* indexes = (int*)PyLong_AsVoidPtr(pyidxs);
            unsigned int count = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,2));
            rtn = tptr->gather(indexes,count);
        }
        else
        {
            //printf("%s\n",cname.c_str());
            PyErr_Format(PyExc_TypeError,"nvstrings: unknown type %s",cname.c_str());
        }
    }
    catch(const std::out_of_range& eor)
    {
        PyErr_Format(PyExc_IndexError,"one or more indexes out of range [0:%u)",tptr->size());
    }
    //
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

//
static PyObject* n_sublist( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int start = 0, step = 1, end = tptr->size();
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        start = (unsigned int)PyLong_AsLong(argOpt);
    argOpt = PyTuple_GetItem(args,2);
    if( argOpt != Py_None )
        end = (unsigned int)PyLong_AsLong(argOpt);
    argOpt = PyTuple_GetItem(args,3);
    if( argOpt != Py_None )
        step = (unsigned int)PyLong_AsLong(argOpt);
    //
    NVStrings* rtn = tptr->sublist(start,end,step);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

//
static PyObject* n_remove_strings( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pyidxs = PyTuple_GetItem(args,1);
    std::string cname = pyidxs->ob_type->tp_name;
    NVStrings* rtn = 0;
    if( cname.compare("list")==0 )
    {
        unsigned int count = (unsigned int)PyList_Size(pyidxs);
        unsigned int* indexes = new unsigned int[count];
        for( int idx=0; idx < count; ++idx )
        {
            PyObject* pyidx = PyList_GetItem(pyidxs,idx);
            indexes[idx] = (unsigned int)PyLong_AsLong(pyidx);
        }
        rtn = tptr->remove_strings(indexes,count,false);
        delete indexes;
    }
    else // device pointer is expected
    {
        unsigned int count = 0;
        PyObject *vo=0, *dptr=0;
        if( !parse_args("remove_strings",args,"OOI",&vo,&dptr,&count) )
            Py_RETURN_NONE;
        unsigned int* indexes = (unsigned int*)PyLong_AsVoidPtr(dptr);
        rtn = tptr->remove_strings(indexes,count);
    }
    //
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

//
static PyObject* n_add_strings( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pyarg = PyTuple_GetItem(args,1);
    std::string cname = pyarg->ob_type->tp_name;
    std::vector<NVStrings*> strslist;
    strslist.push_back(tptr);
    if( cname.compare("list")==0 )
    {
        unsigned int count = (unsigned int)PyList_Size(pyarg);
        for( int idx=0; idx < count; ++idx )
        {
            PyObject* pystrs = PyList_GetItem(pyarg,idx);
            NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pystrs,"m_cptr"));
            strslist.push_back(strs);
        }
    }
    else if( cname.compare("nvstrings")==0 )
    {
        NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pyarg,"m_cptr"));
        strslist.push_back(strs);
    }
    //
    NVStrings* rtn = NVStrings::create_from_strings(strslist);
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_copy( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings* rtn = tptr->copy();
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_isalnum( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        tptr->isalnum(devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    tptr->isalnum(rtn,false);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

static PyObject* n_isalpha( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        tptr->isalpha(devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    tptr->isalpha(rtn,false);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

static PyObject* n_isdigit( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        tptr->isdigit(devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    tptr->isdigit(rtn,false);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

static PyObject* n_isspace( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        tptr->isspace(devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    tptr->isspace(rtn,false);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

static PyObject* n_isdecimal( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        tptr->isdecimal(devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    tptr->isdecimal(rtn,false);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

static PyObject* n_isnumeric( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        tptr->isnumeric(devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    tptr->isnumeric(rtn,false);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

static PyObject* n_islower( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        tptr->islower(devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    tptr->islower(rtn,false);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

static PyObject* n_isupper( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        tptr->isupper(devptr);
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    tptr->isupper(rtn,false);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

// Version 0.1, 0.1.1, 0.2, 0.2.1 features
static PyMethodDef s_Methods[] = {
    { "n_createFromHostStrings", n_createFromHostStrings, METH_VARARGS, "" },
    { "n_destroyStrings", n_destroyStrings, METH_VARARGS, "" },
    { "n_createHostStrings", n_createHostStrings, METH_VARARGS, "" },
    { "n_createFromCSV", n_createFromCSV, METH_VARARGS, "" },
    { "n_createFromOffsets", n_createFromOffsets, METH_VARARGS, "" },
    { "n_createFromNVStrings", n_createFromNVStrings, METH_VARARGS, "" },
    { "n_createFromIntegers", n_createFromIntegers, METH_VARARGS, "" },
    { "n_create_offsets", n_create_offsets, METH_VARARGS, "" },
    { "n_size", n_size, METH_VARARGS, "" },
    { "n_hash", n_hash, METH_VARARGS, "" },
    { "n_null_count", n_null_count, METH_VARARGS, "" },
    { "n_copy", n_copy, METH_VARARGS, "" },
    { "n_remove_strings", n_remove_strings, METH_VARARGS, "" },
    { "n_add_strings", n_add_strings, METH_VARARGS, "" },
    { "n_compare", n_compare, METH_VARARGS, "" },
    { "n_stoi", n_stoi, METH_VARARGS, "" },
    { "n_stof", n_stof, METH_VARARGS, "" },
    { "n_htoi", n_htoi, METH_VARARGS, "" },
    { "n_cat", n_cat, METH_VARARGS, "" },
    { "n_split", n_split, METH_VARARGS, "" },
    { "n_rsplit", n_rsplit, METH_VARARGS, "" },
    { "n_partition", n_partition, METH_VARARGS, "" },
    { "n_rpartition", n_rpartition, METH_VARARGS, "" },
    { "n_split_column", n_split_column, METH_VARARGS, "" },
    { "n_rsplit_column", n_rsplit_column, METH_VARARGS, "" },
    { "n_get", n_get, METH_VARARGS, "" },
    { "n_repeat", n_repeat, METH_VARARGS, "" },
    { "n_pad", n_pad, METH_VARARGS, "" },
    { "n_ljust", n_ljust, METH_VARARGS, "" },
    { "n_center", n_center, METH_VARARGS, "" },
    { "n_rjust", n_rjust, METH_VARARGS, "" },
    { "n_wrap", n_wrap, METH_VARARGS, "" },
    { "n_slice", n_slice, METH_VARARGS, "" },
    { "n_slice_from", n_slice_from, METH_VARARGS, "" },
    { "n_slice_replace", n_slice_replace, METH_VARARGS, "" },
    { "n_replace", n_replace, METH_VARARGS, "" },
    { "n_replace_with_backrefs", n_replace_with_backrefs, METH_VARARGS, "" },
    { "n_len", n_len, METH_VARARGS, "" },
    { "n_byte_count", n_byte_count, METH_VARARGS, "" },
    { "n_lstrip", n_lstrip, METH_VARARGS, "" },
    { "n_strip", n_strip, METH_VARARGS, "" },
    { "n_rstrip", n_rstrip, METH_VARARGS, "" },
    { "n_lower", n_lower, METH_VARARGS, "" },
    { "n_upper", n_upper, METH_VARARGS, "" },
    { "n_capitalize", n_capitalize, METH_VARARGS, "" },
    { "n_swapcase", n_swapcase, METH_VARARGS, "" },
    { "n_title", n_title, METH_VARARGS, "" },
    { "n_translate", n_translate, METH_VARARGS, "" },
    { "n_join", n_join, METH_VARARGS, "" },
    { "n_zfill", n_zfill, METH_VARARGS, "" },
    { "n_find", n_find, METH_VARARGS, "" },
    { "n_find_from", n_find_from, METH_VARARGS, "" },
    { "n_find_multiple", n_find_multiple, METH_VARARGS, "" },
    { "n_rfind", n_rfind, METH_VARARGS, "" },
    { "n_index", n_index, METH_VARARGS, "" },
    { "n_rindex", n_rindex, METH_VARARGS, "" },
    { "n_rindex", n_rindex, METH_VARARGS, "" },
    { "n_findall", n_findall, METH_VARARGS, "" },
    { "n_findall_column", n_findall_column, METH_VARARGS, "" },
    { "n_contains", n_contains, METH_VARARGS, "" },
    { "n_match", n_match, METH_VARARGS, "" },
    { "n_count", n_count, METH_VARARGS, "" },
    { "n_extract", n_extract, METH_VARARGS, "" },
    { "n_extract_column", n_extract_column, METH_VARARGS, "" },
    { "n_startswith", n_startswith, METH_VARARGS, "" },
    { "n_endswith", n_endswith, METH_VARARGS, "" },
    { "n_sort", n_sort, METH_VARARGS, "" },
    { "n_order", n_order, METH_VARARGS, "" },
    { "n_gather", n_gather, METH_VARARGS, "" },
    { "n_sublist", n_sublist, METH_VARARGS, "" },
    { "n_isalnum", n_isalnum, METH_VARARGS, "" },
    { "n_isalpha", n_isalpha, METH_VARARGS, "" },
    { "n_isdigit", n_isdigit, METH_VARARGS, "" },
    { "n_isspace", n_isspace, METH_VARARGS, "" },
    { "n_isdecimal", n_isdecimal, METH_VARARGS, "" },
    { "n_isnumeric", n_isnumeric, METH_VARARGS, "" },
    { "n_islower", n_islower, METH_VARARGS, "" },
    { "n_isupper", n_isupper, METH_VARARGS, "" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef cModPyDem = {	PyModuleDef_HEAD_INIT, "NVStrings_module", "", -1, s_Methods };

PyMODINIT_FUNC PyInit_pyniNVStrings(void)
{
    //NVStrings::initLibrary();
    return PyModule_Create(&cModPyDem);
}
