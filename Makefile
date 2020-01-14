USE_GPU = -DUSE_GPU

CXXFLAGS += -std=c++14 -g -Wall $(USE_GPU) -I$(CUDAINCDIR) -O3 -fopenmp -I${CUBROOT}
LDFLAGS  += -std=c++14 -g -fopenmp 
CXX = c++
CC  = c++

ARCH = -arch=sm_70

NVCC = nvcc
CUBROOT=/mnt/data1/dsr/cub
CUDAFLAGS += -std=c++14 -O3 $(USE_GPU) ${ARCH} -I${CUBROOT}
CUDALDFLAGS += -lcudart -lgomp

strip-cluster : strip-cluster.o Clusterizer.o FEDChannel.o SiStripConditions.o FEDRawData.o SiStripFEDBuffer.o unpackGPU.o
	${NVCC} ${CUDAFLAGS} -o $@ $+ ${CUDALDFLAGS}

strip-cluster.o: strip-cluster.cc Clusterizer.h FEDChannel.h FEDZSChannelUnpacker.h SiStripConditions.h SiStripFEDBuffer.h unpackGPU.cuh
Clusterizer.o: Clusterizer.cc Clusterizer.h SiStripConditions.h
FEDChannel.o : FEDChannel.cc FEDChannel.h Clusterizer.h
SiStripConditions.o : SiStripConditions.cc SiStripConditions.h
FEDRawData.o : FEDRawData.cc FEDRawData.h
SiStripFEDBuffer.o : SiStripFEDBuffer.cc SiStripFEDBuffer.h FEDChannel.h

unpackGPU.o : unpackGPU.cu unpackGPU.cuh
	${NVCC} ${CUDAFLAGS} -c -o $@ $<

clean:
	rm *.o strip-cluster
