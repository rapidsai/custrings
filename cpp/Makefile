#
# Build the C++ interface without using CMake
# This is for experimental purposes and may not create a set of libraries that are compatible with cudf
#
BUILD=build
GENCODE=-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70
NVCFLAGS=-std=c++14 --expt-extended-lambda -Iinclude -I$(RMM_ROOT) -Xcompiler -fPIC -rdc=true -x cu
NVLFLAGS=-L. -L/usr/local/cuda/lib64 -lcudart -lrt
RPATH=-Wl,-rpath,.

all: libNVStrings.so libNVCategory.so libNVText.so

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

NVStrings.o: src/NVStrings.cu include/NVStrings.h src/NVStringsImpl.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVStringsImpl.o: src/NVStringsImpl.cu include/NVStrings.h src/NVStringsImpl.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVStrings_array.o: src/NVStrings_array.cu include/NVStrings.h src/NVStringsImpl.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVStrings_attrs.o: src/NVStrings_attrs.cu include/NVStrings.h src/NVStringsImpl.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVStrings_case.o: src/NVStrings_case.cu include/NVStrings.h src/NVStringsImpl.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVStrings_combine.o: src/NVStrings_combine.cu include/NVStrings.h src/NVStringsImpl.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVStrings_convert.o: src/NVStrings_convert.cu include/NVStrings.h src/NVStringsImpl.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVStrings_find.o: src/NVStrings_find.cu include/NVStrings.h src/NVStringsImpl.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVStrings_modify.o: src/NVStrings_modify.cu include/NVStrings.h src/NVStringsImpl.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVStrings_pad.o: src/NVStrings_pad.cu include/NVStrings.h src/NVStringsImpl.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVStrings_split.o: src/NVStrings_split.cu include/NVStrings.h src/NVStringsImpl.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVStrings_strip.o: src/NVStrings_strip.cu include/NVStrings.h src/NVStringsImpl.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVStrings_substr.o: src/NVStrings_substr.cu include/NVStrings.h src/NVStringsImpl.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVCategory.o: src/NVCategory.cu include/NVCategory.h include/NVStrings.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

NVText.o: src/NVText.cu include/NVText.h include/NVStrings.h src/custring_view.cuh src/custring.cuh
	nvcc $(NVCFLAGS) $(GENCODE) -dc $< -o $@

link_nvstrings.o: NVStrings.o util.o custring_view.o custring.o regexec.o regcomp.o NVStringsImpl.o NVStrings_array.o NVStrings_attrs.o NVStrings_case.o NVStrings_combine.o NVStrings_combine.o NVStrings_convert.o NVStrings_find.o NVStrings_modify.o NVStrings_split.o NVStrings_strip.o NVStrings_substr.o
	nvcc -Xcompiler -fPIC $(GENCODE) -dlink $^ -o $@

link_nvcategory.o: NVCategory.o custring_view.o custring.o
	nvcc -Xcompiler -fPIC $(GENCODE) -dlink $^ -o $@

link_nvtext.o: NVText.o custring_view.o custring.o
	nvcc -Xcompiler -fPIC $(GENCODE) -dlink $^ -o $@

libNVStrings.so: NVStrings.o custring_view.o custring.o util.o regexec.o regcomp.o link_nvstrings.o
	@mkdir -p $(BUILD)
	g++ -shared -fPIC $^ $(NVLFLAGS) -Lbuild -lrmm -o $@ -Wl,-soname,$@ $(RPATH)
	cp $@ $(BUILD)/.

libNVCategory.so: NVCategory.o custring_view.o custring.o link_nvcategory.o libNVStrings.so
	g++ -shared -fPIC $^ $(NVLFLAGS) -o $@ -Wl,-soname,$@ $(RPATH)
	cp $@ $(BUILD)/.

libNVText.so: NVText.o custring_view.o custring.o link_nvtext.o libNVStrings.so
	g++ -shared -fPIC $^ $(NVLFLAGS) -o $@ -Wl,-soname,$@ $(RPATH)
	cp $@ $(BUILD)/.


.PHONY: clean

clean:
	rm -f *.so *.a *.o $(BUILD)/*.so
