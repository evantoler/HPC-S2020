CXX = nvcc
CXXFLAGS = -std=c++11 -O3  -Xcompiler -fopenmp

TARGETS = $(basename $(wildcard *.cu))

all : $(TARGETS)

%:%.cu *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
