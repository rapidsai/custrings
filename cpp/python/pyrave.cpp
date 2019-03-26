/*
* Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <Python.h>
#include <vector>
#include <string>
#include <stdio.h>
#include "NVStrings.h"
#include "util.h"
#include "Rave.h"

// utility to deference NVStrings instance from nvstrings instance
// caller should never dextroy the return object
NVStrings* strings_from_object(PyObject* pystrs)
{
    if( pystrs == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"rave: parameter required");
        return 0;
    }
    std::string cname = pystrs->ob_type->tp_name;
    if( cname.compare("nvstrings")!=0 )
    {
        PyErr_Format(PyExc_ValueError,"rave: argument must be nvstrings object");
        return 0;
    }
    NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pystrs,"m_cptr"));
    if( strs==0 )
        PyErr_Format(PyExc_ValueError,"rave: invalid nvstrings object");
    return strs;
}

// utility to create NVStrings instance from a list of strings
// caller must destroy the returned object
NVStrings* strings_from_list(PyObject* listObj)
{
    unsigned int count = (unsigned int)PyList_Size(listObj);
    if( count==0 )
        return 0;
    //
    const char** list = new const char*[count];
    for( unsigned int idx=0; idx < count; ++idx )
    {
        PyObject* pystr = PyList_GetItem(listObj,idx);
        if( (pystr == Py_None) || !PyObject_TypeCheck(pystr,&PyUnicode_Type) )
            list[idx] = 0;
        else
            list[idx] = PyUnicode_AsUTF8(pystr);
    }
    NVStrings* strs = NVStrings::create_from_array(list,count);
    delete list;
    return strs;
}

//
//
static PyObject* n_unique_tokens( PyObject* self, PyObject* args )
{
    PyObject* pystrs = PyTuple_GetItem(args,0); // only one parm expected
    NVStrings* strs = strings_from_object(pystrs);
    if( strs==0 )
        Py_RETURN_NONE;

    const char* delimiter = " ";
    PyObject* argDelim = PyTuple_GetItem(args,1);
    if( argDelim != Py_None )
        delimiter = PyUnicode_AsUTF8(argDelim);

    strs = Rave::unique_tokens(*strs,delimiter);
    if( strs==0 )
        Py_RETURN_NONE;
    return PyLong_FromVoidPtr((void*)strs);
}

//
static PyObject* n_token_count( PyObject* self, PyObject* args )
{
    PyObject* pystrs = PyTuple_GetItem(args,0);
    NVStrings* strs = strings_from_object(pystrs);
    if( strs==0 )
        Py_RETURN_NONE;

    const char* delimiter = " ";
    PyObject* argDelim = PyTuple_GetItem(args,1);
    if( argDelim != Py_None )
        delimiter = PyUnicode_AsUTF8(argDelim);

    unsigned int* devptr = (unsigned int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    if( devptr )
    {
        unsigned int rtn = Rave::token_count(*strs,delimiter,devptr);
        return PyLong_FromLong((long)rtn);
    }
    //
    unsigned int count = strs->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    unsigned int* rtn = new unsigned int[count];
    Rave::token_count(*strs,delimiter,rtn,false);
    for(unsigned int idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyLong_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

//
static PyObject* n_contains_strings( PyObject* self, PyObject* args )
{
    PyObject* pystrs = PyTuple_GetItem(args,0);
    NVStrings* strs = strings_from_object(pystrs);
    if( strs==0 )
        Py_RETURN_NONE;

    PyObject* argStrs = PyTuple_GetItem(args,1);
    if( argStrs == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"tgts argument must be specified");
        Py_RETURN_NONE;
    }
    NVStrings* tgts = 0;
    std::string cname = argStrs->ob_type->tp_name;
    if( cname.compare("nvstrings")==0 )
        tgts = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(argStrs,"m_cptr"));
    else if( cname.compare("list")==0 )
        tgts = strings_from_list(argStrs);
    //
    if( !tgts )
    {
        PyErr_Format(PyExc_ValueError,"invalid tgts parameter");
        Py_RETURN_NONE;
    }
    if( tgts->size()==0 )
    {
        if( cname.compare("list")==0 )
            NVStrings::destroy(tgts);
        PyErr_Format(PyExc_ValueError,"tgts argument is empty");
        Py_RETURN_NONE;
    }

    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    if( devptr )
    {
        unsigned int rtn = Rave::contains_strings(*strs,*tgts,devptr);
        if( cname.compare("list")==0 )
            NVStrings::destroy(tgts);
        return PyLong_FromLong((long)rtn);
    }
    //
    unsigned int rows = strs->size();
    unsigned int columns = tgts->size();
    PyObject* ret = PyList_New(rows);
    if( rows==0 )
    {
        if( cname.compare("list")==0 )
            NVStrings::destroy(tgts);
        return ret;
    }
    bool* rtn = new bool[rows*columns];
    Rave::contains_strings(*strs,*tgts,rtn,false);
    for(unsigned int idx=0; idx < rows; idx++)
    {
        PyObject* row = PyList_New(columns);
        for( unsigned int jdx=0; jdx < columns; ++jdx )
            PyList_SetItem(row, jdx, PyBool_FromLong((long)rtn[(idx*columns)+jdx]));
        PyList_SetItem(ret, idx, row);
    }
    delete rtn;
    if( cname.compare("list")==0 )
        NVStrings::destroy(tgts);
    return ret;
}

static PyObject* n_strings_counts( PyObject* self, PyObject* args )
{
    PyObject* pystrs = PyTuple_GetItem(args,0);
    NVStrings* strs = strings_from_object(pystrs);
    if( strs==0 )
        Py_RETURN_NONE;
    //
    PyObject* argStrs = PyTuple_GetItem(args,1);
    if( argStrs == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"tgts argument must be specified");
        Py_RETURN_NONE;
    }
    NVStrings* tgts = 0;
    std::string cname = argStrs->ob_type->tp_name;
    if( cname.compare("nvstrings")==0 )
        tgts = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(argStrs,"m_cptr"));
    else if( cname.compare("list")==0 )
        tgts = strings_from_list(argStrs);
    //
    if( !tgts )
    {
        PyErr_Format(PyExc_ValueError,"invalid tgts parameter");
        Py_RETURN_NONE;
    }
    if( tgts->size()==0 )
    {
        if( cname.compare("list")==0 )
            NVStrings::destroy(tgts);
        PyErr_Format(PyExc_ValueError,"tgts argument is empty");
        Py_RETURN_NONE;
    }

    // fill in devptr with result if provided
    unsigned int* devptr = (unsigned int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    if( devptr )
    {
        unsigned int rtn = Rave::strings_counts(*strs,*tgts,devptr);
        if( cname.compare("list")==0 )
            NVStrings::destroy(tgts);
        return PyLong_FromLong((long)rtn);
    }
    // or fill in python list with host memory
    unsigned int rows = strs->size();
    unsigned int columns = tgts->size();
    PyObject* ret = PyList_New(rows);
    if( rows==0 )
    {
        if( cname.compare("list")==0 )
            NVStrings::destroy(tgts);
        return ret;
    }
    unsigned int* rtn = new unsigned int[rows*columns];
    Rave::strings_counts(*strs,*tgts,rtn,false);
    for(unsigned int idx=0; idx < rows; idx++)
    {
        PyObject* row = PyList_New(columns);
        for( unsigned int jdx=0; jdx < columns; ++jdx )
            PyList_SetItem(row, jdx, PyLong_FromLong((long)rtn[(idx*columns)+jdx]));
        PyList_SetItem(ret, idx, row);
    }
    delete rtn;
    if( cname.compare("list")==0 )
        NVStrings::destroy(tgts);
    return ret;
}

static PyObject* n_edit_distance( PyObject* self, PyObject* args )
{
    PyObject* pystrs = PyTuple_GetItem(args,0);
    NVStrings* strs = strings_from_object(pystrs);
    if( strs==0 )
        Py_RETURN_NONE;
    //
    PyObject* pytgts = PyTuple_GetItem(args,1);
    if( pytgts == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"tgt argument must be specified");
        Py_RETURN_NONE;
    }

    unsigned int count = strs->size();
    unsigned int* devptr = (unsigned int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    std::string cname = pytgts->ob_type->tp_name;
    if( cname.compare("str")==0 )
    {
        const char* tgt = PyUnicode_AsUTF8(pytgts);
        if( devptr )
        {
            Rave::edit_distance(Rave::levenshtein,*strs,tgt,devptr);
            Py_RETURN_NONE;
        }
        // or fill in python list with host memory
        PyObject* ret = PyList_New(count);
        if( count==0 )
            return ret;
        std::vector<unsigned int> rtn(count);
        Rave::edit_distance(Rave::levenshtein,*strs,tgt,rtn.data(),false);
        for(unsigned int idx=0; idx < count; idx++)
            PyList_SetItem(ret, idx, PyLong_FromLong((long)rtn[idx]));
        return ret;
    }
    NVStrings* tgts = 0;
    if( cname.compare("nvstrings")==0 )
        tgts = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pytgts,"m_cptr"));
    else if( cname.compare("list")==0 )
        tgts = strings_from_list(pytgts);
    if( !tgts )
    {
        PyErr_Format(PyExc_ValueError,"invalid tgt parameter");
        Py_RETURN_NONE;
    }
    if( tgts->size() != strs->size() )
    {
        PyErr_Format(PyExc_ValueError,"strs and tgt must have the same number of strings");
        Py_RETURN_NONE;
    }
    if( devptr )
    {
        Rave::edit_distance(Rave::levenshtein,*strs,*tgts,devptr);
        if( cname.compare("list")==0 )
            NVStrings::destroy(tgts);
        Py_RETURN_NONE;
    }
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    std::vector<unsigned int> rtn(count);
    Rave::edit_distance(Rave::levenshtein,*strs,*tgts,rtn.data(),false);
    for(unsigned int idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyLong_FromLong((long)rtn[idx]));
    if( cname.compare("list")==0 )
        NVStrings::destroy(tgts);
    return ret;
}

//
static PyMethodDef s_Methods[] = {
    { "n_unique_tokens", n_unique_tokens, METH_VARARGS, "" },
    { "n_token_count", n_token_count, METH_VARARGS, "" },
    { "n_contains_strings", n_contains_strings, METH_VARARGS, "" },
    { "n_strings_counts", n_strings_counts, METH_VARARGS, "" },
    { "n_edit_distance", n_edit_distance, METH_VARARGS, "" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef cModPyDem = {	PyModuleDef_HEAD_INIT, "Rave_module", "", -1, s_Methods };

PyMODINIT_FUNC PyInit_pyniRave(void)
{
    return PyModule_Create(&cModPyDem);
}

