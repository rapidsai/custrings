#
# Documentation on building issues.
# General rules
# - device code from one module cannot be executed within another module (.so)
# - mixing gencode flags causes runtime failures (even when not specified)
# - two link steps are required: one from nvcc (creates new link*.o) and g++ one (creates .so)
#
# NVStrings calls device code in custring and so these must be linked into the same module.
# The libcustring.a is created as a static library so other modules can calls its device methods.
# I think the -rdc=true flag helps with calling device code within the same linked object.
#
# The pyni*.so files do not have any CUDA code and can be linked to libNV*.so files without
# linking to libcustring or any CUDA libraries.
#
# The -Wl,-soname,$@ -Wl,-rpath,.: flags (linker-options) remove the path name from the .so
# and also removes absolute path names for dependent .so files. This is important so that
# the .so files can be found correctly at runtime. Otherwise the location they were found
# during linking will be used at runtime.
#
# Issue found and resolved:
# )) The -lcudart could be changed to -lcudart_static without noticeable effect.
#    This would put static CUDA runtime into every module. This was tested ok with
#    test/* programs like csv.cu and indexes.cpp (compiled with nvcc).
#    However, integrating with csv-reader showed some odd results -- mostly invalid memory addr.
#    But also showed bad kernel launching. These would show up in calls to exclusive_scan where
#    Thrust would thow an exception. Also, cuda-memcheck was helpful in seeing more details.
#    Changing -lcudart_static to -lcudart in this Makefile seemed to fix this.
#    However, csv.cu would no longer work. Building this with nvcc would cause it to now show
#    exceptions in exclusive_scan (from Thrust). The cuda-memcheck showed invalid arguments to
#    the kernel launcher and to cudaPeekLastError (which takes no arguments). Googling this
#    I found that specifying the matching -gencode parameter fixed this issue.
#
# )) The error, "undefined symbol: __fatbinwrap_38_cuda_device_runtime_compute_75",
#    shows up at runtime and sometimes at compile/link time. The solution that has worked
#    so far is to include the -lcudadevrt to the g++ link statement.
#
# )) Still had some exceptions on one 9.0 system when called by the csv-reader and found that
#    adding compute_60 and compute_61 gencode switches seemed to make it work ok.
#
# )) No one likes the rdc=true since it creates extra steps during runtime slowing things down.
#    It also is a viral flag meaning anything that links to such a module must also have rdc=true.
#    Build steps here modified to compile without -rdc=true and without -lcudadevrt.
#    This means no .a files. All device dependencies are now compiled into each .so file.
#

NVCFLAGS=-std=c++14 --expt-extended-lambda -Isrc -Iinclude -Xcompiler -fPIC
NVLFLAGS=-L. -L/usr/local/cuda/lib64 -lcudart -lrt
# may need to include compute_61 when running on 9.0 system
#GENCODE=-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70
GENCODE=-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70

all: libNVStrings.so libNVCategory.so

# The .a files here are for convenience only when prototyping with internal methods
# They are only built if explicitly targeted on the make call (e.g. 'make libcustring.a')
libcustring.a: src/custring_view.cu src/custring_view.cuh src/custring.cu src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -lib -rdc=true src/custring_view.cu src/custring.cu -o $@

libcure.a: src/regex/regexec.cu src/regex/regex.cuh src/regex/regcomp.cpp src/regex/regcomp.h src/custring_view.cuh libcustring.a
	nvcc $(NVCFLAGS) $(GENCODE) -lib -rdc=true $< src/regex/regcomp.cpp libcustring.a -o $@

custring_view.o: src/custring_view.cu src/custring_view.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

custring.o: src/custring.cu src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

regexec.o: src/regex/regexec.cu src/regex/regex.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

regcomp.o: src/regex/regcomp.cpp src/regex/regcomp.h
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

util.o: src/util.cu include/NVStrings.h src/custring_view.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVStrings.o: src/NVStrings.cu include/NVStrings.h src/custring_view.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVCategory.o: src/NVCategory.cu include/NVCategory.h include/NVStrings.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

link_nvstrings.o: NVStrings.o util.o custring_view.o custring.o regexec.o regcomp.o
	nvcc -Xcompiler -fPIC $(GENCODE) -dlink $^ -o $@

link_nvcategory.o: NVCategory.o custring_view.o custring.o
	nvcc -Xcompiler -fPIC $(GENCODE) -dlink $^ -o $@

libNVStrings.so: NVStrings.o custring_view.o custring.o util.o regexec.o regcomp.o link_nvstrings.o
	g++ -shared -fPIC $^ $(NVLFLAGS) -o $@ -Wl,-soname,$@ -Wl,-rpath,.:

libNVCategory.so: NVCategory.o custring_view.o custring.o link_nvcategory.o libNVStrings.so
	g++ -shared -fPIC $^ $(NVLFLAGS) -o $@ -Wl,-soname,$@ -Wl,-rpath,.:

.PHONY: clean install

clean:
	rm -f *.o *.a *.so

install: libNVStrings.so libNVCategory.so NVStrings.h NVCategory.h
	mkdir -p /usr/local/include/custr
	cp NVStrings.h /usr/local/include/custr/.
	cp NVCategory.h /usr/local/include/custr/.
	cp libNVStrings.so /usr/local/lib/.
	cp libNVCategory.so /usr/local/lib/.

uninstall:
	rm -f /usr/local/include/custr/NVStrings.h
	rm -f /usr/local/include/custr/NVCategory.h
	rmdir /usr/local/include/custr
	rm -f /usr/local/lib/libNVStrings.so
	rm -f /usr/local/lib/libNVCategory.so
	
