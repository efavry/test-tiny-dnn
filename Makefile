BASENAME=
# tyny dnn is not compatble with gcc8+ for now
# tiny dnn need a compiler with c++14 functionality
CC=g++

ICC=icl++
CFLAGS=-Wall -Wextra -fpic -Wpedantic -Wno-narrowing -Wno-deprecated -Wno-unused-variable -Wno-unused-parameter -Wno-unused-local-typedefs -std=gnu++14 -I.
DCFLAGS=-g -Wno-unknown-pragmas -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -DDEBUG
RCFLAGS=-O3 -DCNN_USE_AVX -DCNN_USE_SSE -DDNN_USE_IMAGE_API -pthread -msse3 -mavx -DNDEBUG 
#I/home/qsm/Documents/GWU/FisrtPaperIsHope/tiny-dnn-master
#-D CNN_USE_LIBDNN 
#-fopenmp -D USE_OMP
#voir https://github.com/tiny-dnn/tiny-dnn/blob/6ddac1ef3d7ff1594a7ec8c71caa125cc096214f/CMakeLists.txt chercher USE_TBB
#voir https://github.com/tiny-dnn/tiny-dnn/blob/6ddac1ef3d7ff1594a7ec8c71caa125cc096214f/cmake/summary.cmake chercher LibDNN
#voir https://github.com/tiny-dnn/tiny-dnn section buildinf
 
DTARGET=test_debug
RTARGET=test_release

SOURCEFILE=main.cpp

all: release
#$(INCLUDE)

release_icc: 
	$(ICC) $(CFLAGS) $(RCFLAGS) -o ./bin/$(RTARGET) $(SOURCEFILE)
#$@ $^
release: 
	$(CC) $(CFLAGS) $(RCFLAGS) -o ./bin/$(RTARGET) $(SOURCEFILE)

debug: 
	$(CC) $(CFLAGS) $(DCFLAGS) -o ./bin/$(DTARGET) $(SOURCEFILE)

read_test: read_test.cpp
	$(CC) $(CFLAGS) $(RCFLAGS) -o ./bin/$@ $^

predict: predict.cpp
	$(CC) $(CFLAGS) $(RCFLAGS) -o ./bin/$@ $^
	
clean:
	rm -f ../bin/$(DTARGET)
	rm -f ../bin/$(RTARGET)

clean_all:
	rm -f ../bin/$(DTARGET)
	rm -f ../bin/$(RTARGET)
