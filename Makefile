CXXFLAGS += -std=c++14 -g -Wall #-O3 -fopenmp 
LDFLAGS  += -std=c++14 #-fopenmp 
CXX = c++
CC = c++

strip-cluster : strip-cluster.o Clusterizer.o FEDChannel.o SiStripConditions.o FEDRawData.o SiStripFEDBuffer.o
strip-cluster.o: strip-cluster.cc Clusterizer.h FEDChannel.h FEDZSChannelUnpacker.h SiStripConditions.h SiStripFEDBuffer.h
Clusterizer.o: Clusterizer.cc Clusterizer.h SiStripConditions.h
FEDChannel.o : FEDChannel.cc FEDChannel.h Clusterizer.h
SiStripConditions.o : SiStripConditions.cc SiStripConditions.h
FEDRawData.o : FEDRawData.cc FEDRawData.h
SiStripFEDBuffer.o : SiStripFEDBuffer.cc SiStripFEDBuffer.h FEDChannel.h
