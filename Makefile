USE_GPU = -DUSE_GPU

DEFS = $(USE_GPU) -DCACHE_ALLOC -DCALIB_1D

CXXFLAGS += -std=c++14 -g -Wall $(DEFS) -I$(CUDAINCDIR) -O3 -fopenmp -I${CUBROOT}
LDFLAGS  += -std=c++14 -g -fopenmp 
CXX = c++
CC  = c++

ARCH = -arch=sm_70

NVCC = nvcc
CUBROOT=/mnt/data1/dsr/cub
CUDAFLAGS += -std=c++14 -O3 $(DEFS) ${ARCH} -I${CUBROOT}
CUDALDFLAGS += -lcudart -lgomp

strip-cluster : strip-cluster.o Clusterizer.o FEDChannel.o SiStripConditions.o FEDRawData.o SiStripFEDBuffer.o unpackGPU.o clusterGPU.o allocate_host.o  allocate_device.o cluster.o
	${NVCC} ${CUDAFLAGS} -o $@ $+ ${CUDALDFLAGS}

strip-cluster.o: strip-cluster.cc Clusterizer.h FEDChannel.h FEDZSChannelUnpacker.h SiStripConditions.h SiStripFEDBuffer.h unpackGPU.cuh cluster.h
Clusterizer.o: Clusterizer.cc Clusterizer.h SiStripConditions.h
FEDChannel.o : FEDChannel.cc FEDChannel.h Clusterizer.h
SiStripConditions.o : SiStripConditions.cc SiStripConditions.h
FEDRawData.o : FEDRawData.cc FEDRawData.h
SiStripFEDBuffer.o : SiStripFEDBuffer.cc SiStripFEDBuffer.h FEDChannel.h
allocate_host.o: allocate_host.cc getCachingHostAllocator.h CachingHostAllocator.h
allocate_device.o: allocate_device.cc allocate_device.h getCachingDeviceAllocator.h CachingDeviceAllocator.h
cluster.o: cluster.cc cluster.h

unpackGPU.o : unpackGPU.cu unpackGPU.cuh
	${NVCC} ${CUDAFLAGS} -c -o $@ $<

clusterGPU.o : clusterGPU.cu clusterGPU.cuh cluster.h
	${NVCC} ${CUDAFLAGS} -c -o $@ $<

clean:
	rm *.o strip-cluster
