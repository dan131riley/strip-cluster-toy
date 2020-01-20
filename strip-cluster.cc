#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <functional>
#include <chrono>

typedef std::chrono::time_point<std::chrono::system_clock> timepoint;
typedef std::chrono::duration<double> tick;

static timepoint now()
{
  return std::chrono::system_clock::now();
}

static tick delta(timepoint& t0)
{
  timepoint t1(now());
  tick d = t1 - t0;
  t0 = t1;
  return d;
}

#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cub/util_debug.cuh>
#include "cudaCheck.h"
#include "unpackGPU.cuh"
#include "copyAsync.h"
#endif

#include "Clusterizer.h"
#include "FEDRawData.h"
#include "SiStripFEDBuffer.h"
#include "FEDZSChannelUnpacker.h"

//#define DBGPRINT 1

class StripByStripAdder {
public:
  typedef std::output_iterator_tag iterator_category;
  typedef void value_type;
  typedef void difference_type;
  typedef void pointer;
  typedef void reference;

  StripByStripAdder(Clusterizer& clusterizer,
                    Clusterizer::State& state,
                    std::vector<SiStripCluster>& record)
    : clusterizer_(clusterizer), state_(state), record_(record) {}

  StripByStripAdder& operator= ( SiStripDigi digi )
  {
    clusterizer_.stripByStripAdd(state_, digi.strip(), digi.adc(), record_);
    return *this;
  }

  StripByStripAdder& operator*  ()    { return *this; }
  StripByStripAdder& operator++ ()    { return *this; }
  StripByStripAdder& operator++ (int) { return *this; }
private:
  Clusterizer& clusterizer_;
  Clusterizer::State& state_;
  std::vector<SiStripCluster>& record_;
};

void
testUnpackZS(const std::vector<uint8_t>& alldata, 
             const SiStripConditions* conditions, 
             const ChannelLocs& chanlocs)
{
  std::vector<uint16_t> stripId(alldata.size());

#pragma omp parallel for
  for(size_t i = 0; i < chanlocs.size(); ++i) {
    const auto channel = FEDChannel(alldata.data(), chanlocs.offset(i), chanlocs.length(i));

    if (channel.length() > 0) {
      const uint8_t* data = channel.data();
      const auto payloadOffset = channel.offset();
      const auto payloadLength = channel.length();
      auto offset = payloadOffset;

#if defined(DBGPRINT)
      const auto& detmap = conditions->detToFeds();
      const auto& detp = detmap[i];
      const auto fedId = detp.fedID();
      const auto fedCh = detp.fedCh();
      const auto detid = detp.detID();
      const auto ipair = detp.pair();
      std::cout << "FED " << fedId << " channel " << (int) fedCh << " det:pair " << detid << ":" << ipair << std::endl;
      std::cout << "Offset " << payloadOffset << " Length " << payloadLength << std::endl;
#endif

      while (offset < payloadOffset+payloadLength) {
        stripId[offset] = invStrip;
        const auto stripIndex = data[(offset++)];
        stripId[offset] = invStrip;
        const auto groupLength = data[(offset++)];
        for (auto i = 0; i < groupLength; ++i, ++offset) {
          // auto adc = data[offset];
          stripId[offset] = stripIndex + i;
        }
      }
    }
#if defined(DBGPRINT)
    else {
      std::cout << " Index " << i << " length " << channel.length() << std::endl;
    }
#endif
  }
}

template<typename OUT>
OUT unpackZS(const FEDChannel& chan, uint16_t stripOffset, OUT out, detId_t idet)
{
  auto unpacker = FEDZSChannelUnpacker::zeroSuppressedZeroModeUnpacker(chan);
  while (unpacker.hasData()) { *out++ = SiStripDigi(stripOffset+unpacker.sampleNumber(), unpacker.adc()); unpacker++; }
  return out;
}

using SiStripClusters = std::vector<SiStripCluster>;
using SiStripClusterMap = std::map<detId_t, SiStripClusters>;

void printClusters(detId_t idet, const SiStripClusters& clusters)
{
  std::cout << "Printing clusters for detid " << idet << std::endl;

  for (const auto& cluster : clusters) {
    std::cout << "Cluster " << cluster.firstStrip() << ": ";
    for (const auto& ampl : cluster.amplitudes()) {
      std::cout << (int) ampl << " ";
    }
    std::cout << std::endl;
  }
}

SiStripClusterMap
fillClusters(const std::vector<uint8_t>& alldata, 
             const SiStripConditions* conditions, 
             const ChannelLocs& chanlocs)
{
  Clusterizer clusterizer(conditions);
  Clusterizer::State state;
  SiStripClusters out;
  SiStripClusterMap clusters;
  auto prevDet = invDet;
  Clusterizer::Det det(conditions, invFed, 0);

  for(size_t i = 0; i < chanlocs.size(); ++i) {
    const auto chan = FEDChannel(alldata.data(), chanlocs.offset(i), chanlocs.length(i));;

    auto fedId = chanlocs.fedID(i);
    auto fedCh = chanlocs.fedCh(i);
    auto detid = (*conditions)(fedId, fedCh).detID();
    auto ipair = (*conditions)(fedId, fedCh).iPair();

    if (detid != prevDet) {
#if defined(DBGPRINT)
      std::cout << "DetID " << prevDet << " clusters " << out.size() << std::endl;
#endif
      if (out.size() > 0) {
        clusters[prevDet] = std::move(out);
      }
      det = clusterizer.stripByStripBegin(fedId);
      state.reset(det);
      prevDet = detid;
    }

    det.setFedCh(fedCh);
#if defined(DBGPRINT)
    std::cout << "FED " << fedId << " channel " << (int) fedCh << " detid " << detid << " ipair " << ipair 
              << " len:off " << chan.length() << ":" << chan.offset() << std::endl;
#endif

    if (chan.length() > 0) {
      auto perStripAdder = StripByStripAdder(clusterizer, state, out);
      unpackZS(chan, ipair*256, perStripAdder, detid);
    }
  }

  return clusters;
}

int main(int argc, char** argv)
{
  std::string datafilename("stripdata.bin");
  std::string condfilename("stripcond.bin");

  if (argc > 1) {
    std::string prefix(argv[1]);
    datafilename = prefix + datafilename;
    condfilename = prefix + condfilename;
  }
  std::cout << "Reading " << datafilename << "+" << condfilename << std::endl;

  auto conditions = std::make_unique<SiStripConditions>(condfilename);
#ifdef USE_GPU
  int gpu_device = 0;
  cudaCheck(cudaSetDevice(gpu_device));
  cudaCheck(cudaGetDevice(&gpu_device));

  cudaStream_t stream;
  cudaCheck(cudaStreamCreate(&stream));
  
  std::unique_ptr<SiStripConditionsGPU, std::function<void(SiStripConditionsGPU*)>>
    condGPU(conditions->toGPU(), [](SiStripConditionsGPU* p) { cudaFree(p); });
#endif

  std::ifstream datafile(datafilename, std::ios::in | std::ios::binary);
  datafile.seekg(sizeof(size_t)); // skip initial event mark

  std::vector<FEDRawData> fedRawDatav;
  std::vector<fedId_t> fedIdv;
  std::vector<FEDBuffer> fedBufferv;
  std::vector<fedId_t> fedIndex(SiStripConditions::kFedCount);
  std::vector<uint8_t> alldata;

  ChannelLocs chanlocs(conditions->detToFeds().size(), stream);

  fedRawDatav.reserve(SiStripConditions::kFedCount);
  fedIdv.reserve(SiStripConditions::kFedCount);
  fedBufferv.reserve(SiStripConditions::kFedCount);

#ifdef USE_GPU
  std::vector<cudautils::device::unique_ptr<uint8_t[]>> fedRawDataGPU;
  std::vector<uint8_t*> inputGPU(chanlocs.size());
  ChannelLocsGPU chanlocsGPU(chanlocs.size(), stream);

  fedRawDataGPU.reserve(SiStripConditions::kFedCount);
#endif

  while (!datafile.eof()) {
    size_t size = 0;
    size_t totalSize = 0;
    FEDReadoutMode mode = READOUT_MODE_INVALID;

#ifdef USE_GPU
    fedRawDataGPU.clear();
#endif
    fedRawDatav.clear();
    fedIdv.clear();
    fedIndex.clear();
    fedIndex.resize(SiStripConditions::kFedCount, invFed);
    alldata.clear();

    // read in the raw data
    while (datafile.read((char*) &size, sizeof(size)).gcount() == sizeof(size) && size != std::numeric_limits<size_t>::max()) {
      int fedId = 0;
      datafile.read((char*) &fedId, sizeof(fedId));
#if defined(DBGPRINT)
      std::cout << "Reading FEDRawData ID " << fedId << " size " << size << std::endl;
#endif
      fedRawDatav.emplace_back(size, stream);
      auto& rawData = fedRawDatav.back();

      datafile.read((char*) rawData.get(), size);

      fedIndex[fedId-SiStripConditions::kFedFirst] = fedIdv.size();
      fedIdv.push_back(fedId);
      
#ifdef USE_GPU
      auto tmp = cudautils::make_device_unique<uint8_t[]>(size, stream);
      cudautils::copyAsync(tmp, rawData.data(), size, stream);
      fedRawDataGPU.emplace_back(tmp.release());
#endif

      fedBufferv.emplace_back(rawData.get(), rawData.size());

      if (fedBufferv.size() == 1) {
        mode = fedBufferv.back().readoutMode();
      } else {
        assert(fedBufferv.back().readoutMode() == mode);
      }

      totalSize += size;
    }

    const auto& detmap = conditions->detToFeds();
    size_t offset = 0;

    // iterate over the detector in DetID/APVPair order
    // mapping out where the data are
    const uint16_t headerlen = mode == READOUT_MODE_ZERO_SUPPRESSED ? 7 : 2;

    for(size_t i = 0; i < detmap.size(); ++i) {
      const auto& detp = detmap[i];

      auto fedId = detp.fedID();
      auto fedi = fedIndex[fedId-SiStripConditions::kFedFirst];
      if (fedi != invFed) {
        const auto& buffer = fedBufferv[fedi];
        const auto& channel = buffer.channel(detp.fedCh());

        if (channel.length() >= headerlen) {
          chanlocs.setChannelLoc(i, channel.data(), channel.offset()+headerlen, offset, channel.length()-headerlen,
                                 detp.fedID(), detp.fedCh());
          offset += channel.length()-headerlen;
        } else {
          chanlocs.setChannelLoc(i, channel.data(), channel.offset(), offset, channel.length(),
                                 detp.fedID(), detp.fedCh());
          offset += channel.length();
          assert(channel.length() == 0);
        }
#ifdef USE_GPU
        inputGPU[i] = fedRawDataGPU[fedi].get() + (channel.data() - fedRawDatav[fedi].get());
#endif
      } else {
        chanlocs.setChannelLoc(i, nullptr, 0, 0, 0, invFed, 0);
#ifdef USE_GPU
        inputGPU[i] = nullptr;
#endif
        std::cout << "Missing fed " << fedi << " for detID " << detp.fedID() << std::endl;
      }
    }

    std::cout << "Total size " << totalSize << " channel sum " << offset << std::endl;
    alldata.resize(offset); // resize to the amount of data

#ifdef USE_GPU
    chanlocsGPU.reset(chanlocs, inputGPU, stream);
    StripDataGPU stripdata(offset, stream);

    timepoint t0(now());
    unpackChannelsGPU(chanlocsGPU, condGPU.get(), stripdata, stream);

    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    CubDebugExit(e);
    tick GPUtime = delta(t0);

    auto outdata = cudautils::make_host_unique<uint8_t[]>(offset, stream);
    cudautils::copyAsync(outdata, stripdata.alldataGPU_, offset, stream);
#endif

    // iterate over the detector in DetID/APVPair order
    // copying the data into the alldata array
    // This could be combined with the previous loop, but
    // this loop can be parallelized, previous is serial
    timepoint t1(now());

#pragma omp parallel for
    for(size_t i = 0; i < chanlocs.size(); ++i) {
      const auto data = chanlocs.input(i);

      if (data != nullptr) {
        auto aoff = chanlocs.offset(i);
        auto choff = chanlocs.inoff(i);

        for (auto k = 0; k < chanlocs.length(i); ++k) {
          alldata[aoff] = data[choff^7];
#if defined(USE_GPU)
          assert(alldata[aoff] == outdata[aoff]);
#endif
          aoff++; choff++;
        }
      }
    }
    testUnpackZS(alldata, conditions.get(), chanlocs);
    tick CPUtime = delta(t1);

    std::cout << "Times GPU/CPU " << GPUtime.count() << "/" << CPUtime.count() << std::endl;

    auto clusters = fillClusters(alldata, conditions.get(), chanlocs);

    const detId_t idet = 369120277;
    printClusters(idet, clusters[idet]);
  }
}
