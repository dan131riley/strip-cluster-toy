CXXFLAGS += -std=c++14 -g -Wall -DUSE_GPU -I$(CUDAINCDIR) #-O3 -fopenmp 
LDFLAGS  += -std=c++14 -g #-fopenmp 
CXX = c++
CC  = c++

ARCH = -arch=sm_70

NVCC = nvcc
CUBROOT=/mnt/data1/dsr/cub
CUDAFLAGS += -std=c++14 -O3 ${ARCH} -I${CUBROOT}
CUDALDFLAGS += -lcudart

strip-cluster : strip-cluster.o Clusterizer.o FEDChannel.o SiStripConditions.o FEDRawData.o SiStripFEDBuffer.o
	${NVCC} ${CUDAFLAGS} -o $@ $+ ${CUDALDFLAGS}

strip-cluster.o: strip-cluster.cc Clusterizer.h FEDChannel.h FEDZSChannelUnpacker.h SiStripConditions.h SiStripFEDBuffer.h
Clusterizer.o: Clusterizer.cc Clusterizer.h SiStripConditions.h
FEDChannel.o : FEDChannel.cc FEDChannel.h Clusterizer.h
SiStripConditions.o : SiStripConditions.cc SiStripConditions.h
FEDRawData.o : FEDRawData.cc FEDRawData.h
SiStripFEDBuffer.o : SiStripFEDBuffer.cc SiStripFEDBuffer.h FEDChannel.h

clean:
	rm *.o strip-cluster
