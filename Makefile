# tiny dnn is not compatble with gcc8+ for now
# tiny dnn need a compiler with c++14 functionality
CC=g++

ICC=icc++
CFLAGS=-Wall -Wextra -fpic -Wpedantic -Wno-narrowing -Wno-deprecated -Wno-unused-variable -Wno-unused-parameter -Wno-unused-local-typedefs -std=gnu++14 -I.
DCFLAGS=-g -Wno-unknown-pragmas -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -DDEBUG
RCFLAGS=-O3 -DDNN_USE_IMAGE_API -pthread  -DNDEBUG 
 
DTARGET=test_debug
RTARGET=test_release

SOURCEFILE=main.cpp

all: release

release_icc: 
	$(ICC) $(CFLAGS) $(RCFLAGS) -o ./bin/$(RTARGET) $(SOURCEFILE)
release: 
	$(CC) $(CFLAGS) $(RCFLAGS) -o ./bin/$(RTARGET) $(SOURCEFILE)

debug: 
	$(CC) $(CFLAGS) $(DCFLAGS) -o ./bin/$(DTARGET) $(SOURCEFILE)

read_test: read_test.cpp
	$(CC) $(CFLAGS) $(RCFLAGS) -o ./bin/$@ $^

predict: predict.cpp
	$(CC) $(CFLAGS) $(RCFLAGS) -o ./bin/$@ $^

predict_measure: predict.cpp
	$(CC) $(CFLAGS) $(RCFLAGS) -DMEASURE -o ./bin/$@ $^
	
clean:
	rm -f ../bin/$(DTARGET)
	rm -f ../bin/$(RTARGET)

clean_all:
	rm -f ../bin/$(DTARGET)
	rm -f ../bin/$(RTARGET)
