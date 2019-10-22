#include <fstream>
#include <iostream>
#include <algorithm>

#include "Clusterizer.h"
#include "FEDRawData.h"
#include "SiStripFEDBuffer.h"
#include "FEDZSChannelUnpacker.h"

#define DBGPRINT 1

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
testUnpackZS(int fedId, const FEDBuffer& buffer, const SiStripConditions* conditions)
{
  std::vector<uint16_t> stripId(buffer.bufferSize());

  const uint16_t headerlen = buffer.readoutMode() == READOUT_MODE_ZERO_SUPPRESSED ? 7 : 2;

  //#pragma omp for
  for (auto fedCh = 0; fedCh < buffer.validChannels(); ++fedCh) {
    const auto& channel = buffer.channel(fedCh);

    if (channel.length() > 0) {
      const uint8_t* data = channel.data();
      uint16_t payloadOffset = channel.offset()+headerlen;
      uint16_t offset = payloadOffset;
      uint16_t payloadLength = channel.length()-headerlen;

  #ifdef DBGPRINT
      auto ipair = (*conditions)(fedId, fedCh).iPair();
      auto detid = (*conditions)(fedId, fedCh).detID();
      std::cout << "FED " << fedId << " channel " << fedCh << " det:pair " << detid << ":" << ipair << std::endl;
      std::cout << "Offset " << payloadOffset << " Length " << payloadLength << std::endl;
  #endif

      for (auto i = channel.offset(); i < channel.offset() + headerlen; ++i) {
        stripId[i^7] = invStrip;
      }

      while (offset < payloadOffset+payloadLength) {
        stripId[offset^7] = invStrip;
        uint8_t stripIndex = data[(offset++)^7];
        stripId[offset^7] = invStrip;
        uint8_t groupLength = data[(offset++)^7];
        for (auto i = 0; i < groupLength; ++i, ++offset) {
          // auto adc = data[offset^7];
          stripId[offset^7] = stripIndex + i;
        }
      }
    }
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
fillClusters(int fedId, const SiStripConditions* conditions, const std::vector<FEDChannel>& channels, const FEDReadoutMode mode)
{
  Clusterizer clusterizer(conditions);
  Clusterizer::State state;
  SiStripClusters out;
  SiStripClusterMap clusters;
  detId_t prevDet = invDet;
  Clusterizer::Det det(conditions, fedId);

  for (auto fedCh = 0; fedCh < channels.size(); ++fedCh) {
    auto ipair = (*conditions)(fedId, fedCh).iPair();
    auto detid = (*conditions)(fedId, fedCh).detID();

    if (detid != prevDet) {
#ifdef DBGPRINT
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
    const auto& chan = channels[fedCh];
#ifdef DBGPRINT
    std::cout << "FED " << fedId << " channel " << fedCh << " detid " << detid << " ipair " << ipair 
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

  auto conditions = std::make_unique<SiStripConditions>("stripcond.bin");

  std::ifstream datafile(datafilename, std::ios::in | std::ios::binary);
  FEDRawData rawData;

  datafile.seekg(sizeof(size_t)); // skip initial event mark

  while (!datafile.eof()) {
    size_t size = 0;
    while (datafile.read((char*) &size, sizeof(size)).gcount() == sizeof(size) && size != std::numeric_limits<size_t>::max()) {
      int fedId = 0;
      datafile.read((char*) &fedId, sizeof(fedId));
#ifdef DBGPRINT
      std::cout << "Reading FEDRawData ID " << fedId << " size " << size << std::endl;
#endif
      rawData.resize(size);
      datafile.read((char*) rawData.data(), size);
      FEDBuffer buffer(rawData.data(),rawData.size());

      const FEDReadoutMode mode = buffer.readoutMode();

      testUnpackZS(fedId, buffer, conditions.get());
      auto clusters = fillClusters(fedId, conditions.get(), buffer.channels(), mode);

      if (fedId == 50) {
        const detId_t idet = 369120277;
        printClusters(idet, clusters[idet]);
      }
    }
  }
}
