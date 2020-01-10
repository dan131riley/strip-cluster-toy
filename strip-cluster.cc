#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <functional>

#ifdef USE_GPU
#include <cuda_runtime.h>
#include "cuda_rt_call.h"
#include "unpackGPU.cuh"
#endif

#include "Clusterizer.h"
#include "FEDRawData.h"
#include "SiStripFEDBuffer.h"
#include "FEDZSChannelUnpacker.h"

//#define DBGPRINT 1
#define ALIGN 16
#if defined(ALIGN)
static constexpr auto kAlign = ALIGN;
#endif

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
             const ChannelLocs& chanlocs,
             const FEDReadoutMode mode)
{
  std::vector<uint16_t> stripId(alldata.size());

  const uint16_t headerlen = mode == READOUT_MODE_ZERO_SUPPRESSED ? 7 : 2;

#pragma omp for
  for(size_t i = 0; i < chanlocs.size(); ++i) {
    const auto channel = FEDChannel(alldata.data(), chanlocs.offset(i), chanlocs.length(i));

    if (channel.length() > 0) {
      const uint8_t* data = channel.data();
      const auto payloadOffset = channel.offset()+headerlen;
      const auto payloadLength = channel.length()-headerlen;
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

      for (auto i = channel.offset(); i < channel.offset() + headerlen; ++i) {
        stripId[i] = invStrip;
      }

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
OUT unpackZS(const FEDChannel& chan, FEDReadoutMode mode, uint16_t stripOffset, OUT out, detId_t idet)
{
  switch ( mode ) {
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8:
    {
      auto unpacker = FEDZSChannelUnpacker::zeroSuppressedLiteModeUnpacker(chan);
      while (unpacker.hasData()) { *out++ = SiStripDigi(stripOffset+unpacker.sampleNumber(), unpacker.adc()); unpacker++; }
    }
    break;
    case READOUT_MODE_ZERO_SUPPRESSED:
    {
      auto unpacker = FEDZSChannelUnpacker::zeroSuppressedModeUnpacker(chan);
      while (unpacker.hasData()) { *out++ = SiStripDigi(stripOffset+unpacker.sampleNumber(), unpacker.adc()); unpacker++; }
    }
    break;
    default:
      ::abort();
  }

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
             const ChannelLocs& chanlocs,
             const FEDReadoutMode mode)
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
      unpackZS(chan, mode, ipair*256, perStripAdder, detid);
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
  std::unique_ptr<SiStripConditionsGPU, std::function<void(SiStripConditionsGPU*)>>
    condGPU(conditions->toGPU(), [](SiStripConditionsGPU* p) { cudaFree(p); });
#endif

  std::ifstream datafile(datafilename, std::ios::in | std::ios::binary);
  datafile.seekg(sizeof(size_t)); // skip initial event mark

  FEDRawData rawData;
  std::vector<FEDRawData> fedRawDatav;
  std::vector<fedId_t> fedIdv;
  std::vector<FEDBuffer> fedBufferv;
  std::vector<fedId_t> fedIndex(SiStripConditions::kFedCount);
  std::vector<uint8_t> alldata;

  ChannelLocs chanlocs(conditions->detToFeds().size());

  fedRawDatav.reserve(SiStripConditions::kFedCount);
  fedIdv.reserve(SiStripConditions::kFedCount);
  fedBufferv.reserve(SiStripConditions::kFedCount);

#ifdef USE_GPU
  std::vector<uint8_t*> fedRawDataGPU;
  std::vector<uint8_t*> inputGPU(chanlocs.size());
  ChannelLocsGPU chanlocsGPU(chanlocs.size());

  fedRawDataGPU.reserve(SiStripConditions::kFedCount);
#endif

  while (!datafile.eof()) {
    size_t size = 0;
    size_t totalSize = 0;
    FEDReadoutMode mode;

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
      rawData.resize(size);
      datafile.read((char*) rawData.data(), size);

      fedIndex[fedId-SiStripConditions::kFedFirst] = fedIdv.size();
      fedIdv.push_back(fedId);
      auto addr = rawData.data();
      fedRawDatav.push_back(std::move(rawData));
      
      const auto& rd = fedRawDatav.back();
      assert(rd.data() == addr);
      fedBufferv.emplace_back(rd.data(), rd.size());

      if (fedBufferv.size() == 1) {
        mode = fedBufferv.back().readoutMode();
      } else {
        assert(fedBufferv.back().readoutMode() == mode);
      }

      totalSize += size;
#ifdef USE_GPU
      uint8_t* tmp;
      CUDA_RT_CALL(cudaMalloc((void**) &tmp, sizeof(uint8_t)*size));
      CUDA_RT_CALL(cudaMemcpyAsync(tmp, rd.data(), sizeof(uint8_t)*size, cudaMemcpyDefault));
      fedRawDataGPU.push_back(tmp);
#endif
    }

    const auto& detmap = conditions->detToFeds();
    size_t offset = 0;

    // iterate over the detector in DetID/APVPair order
    // mapping out where the data are
    for(size_t i = 0; i < detmap.size(); ++i) {
      const auto& detp = detmap[i];

      auto fedId = detp.fedID();
      auto fedi = fedIndex[fedId-SiStripConditions::kFedFirst];
      if (fedi != invFed) {
        const auto& buffer = fedBufferv[fedi];
        const auto& channel = buffer.channel(detp.fedCh());
        chanlocs.setChannelLoc(i, channel.data(), channel.offset(), offset, channel.length(), detp.fedID(), detp.fedCh());
#ifdef USE_GPU
        inputGPU[i] = fedRawDataGPU[fedi] + (channel.data() - fedRawDatav[fedi].data());
#endif
#if defined(ALIGN)
        offset += kAlign*((channel.length()-1)/kAlign + 1); // alignment
#else
        offset += channel.length();
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
    chanlocsGPU.reset(chanlocs, inputGPU);
    uint8_t* alldataGPU;
    detId_t* detIdGPU;
    stripId_t* stripIdGPU;
    CUDA_RT_CALL(cudaMalloc((void**) &alldataGPU, sizeof(uint8_t)*offset));
    CUDA_RT_CALL(cudaMalloc((void**) &detIdGPU, sizeof(detId_t)*offset));
    CUDA_RT_CALL(cudaMalloc((void**) &stripIdGPU, sizeof(stripId_t)*offset));

    unpackChannelsGPU(chanlocsGPU, condGPU.get(), alldataGPU, detIdGPU, stripIdGPU);

    std::vector<uint8_t> outdata(offset);
    CUDA_RT_CALL(cudaMemcpy(outdata.data(), alldataGPU, sizeof(uint8_t)*offset, cudaMemcpyDefault));
#endif

    // iterate over the detector in DetID/APVPair order
    // copying the data into the alldata array
    // This could be combined with the previous loop, but
    // this loop can be parallelized, previous is serial
#pragma omp for
    for(size_t i = 0; i < chanlocs.size(); ++i) {
      const auto data = chanlocs.input(i);

      if (data != nullptr) {
        auto aoff = chanlocs.offset(i);
        auto choff = chanlocs.inoff(i);

        for (auto k = 0; k < chanlocs.length(i); ++k) {
          alldata[aoff] = data[choff^7];
          if (i == 0 && k < 8) {
            printf("Offset %lu/%lu data 0x%02x/0x%02x\n", choff^7, aoff, (int) data[choff^7], (int) alldata[aoff]);
          }
          assert(alldata[aoff] == outdata[aoff]);
          aoff++; choff++;
        }
      }
    }

    testUnpackZS(alldata, conditions.get(), chanlocs, mode);
    auto clusters = fillClusters(alldata, conditions.get(), chanlocs, mode);

    const detId_t idet = 369120277;
    printClusters(idet, clusters[idet]);
#ifdef USE_GPU
    cudaFree(alldataGPU);
    cudaFree(detIdGPU);
    cudaFree(stripIdGPU);
    for (auto m : fedRawDataGPU) {
      cudaFree(m);
    }
#endif
  }
}
