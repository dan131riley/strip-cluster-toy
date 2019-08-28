#include <fstream>
#include <iostream>
#include <algorithm>

#include "Clusterizer.h"
#include "FEDZSChannelUnpacker.h"

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

void testUnpackZS(const FEDChannel& channel, uint16_t stripOffset)
{
  const uint8_t* data = channel.data()-channel.offset();
  uint16_t payloadOffset = channel.offset()+7;
  uint16_t offset = payloadOffset;
  uint16_t payloadLength = channel.length()-7;

  while (offset < payloadOffset+payloadLength) {
    uint8_t stripIndex = data[(offset++)^7];
    uint8_t groupLength = data[(offset++)^7];
    std::cout << "New group offset " << offset << " length " << (int) groupLength << " first channel " << (int) stripIndex << std::endl;
    offset += groupLength;
  }
}


template<typename OUT>
OUT unpackZS(const FEDChannel& chan, uint16_t stripOffset, OUT out, detId_t idet)
{
  auto unpacker = FEDZSChannelUnpacker::zeroSuppressedModeUnpacker(chan);
  while (unpacker.hasData()) {
    auto digi = SiStripDigi(stripOffset+unpacker.sampleNumber(), unpacker.adc());
    if (digi.strip() != 0) {
      *out++ = digi;
    }
    unpacker++;
  }
  return out;
}

FEDSet fillFeds()
{
  std::ifstream fedfile("stripdata.bin", std::ios::in | std::ios::binary);

  FEDSet feds;
  detId_t detid;  

  while (fedfile.read((char*)&detid, sizeof(detid)).gcount() == sizeof(detid)) {
    FEDChannel fed(fedfile);
    feds[detid].push_back(std::move(fed));
  }
  return feds;
}

std::vector<SiStripCluster>
fillClusters(detId_t idet, Clusterizer& clusterizer, Clusterizer::State& state, const std::vector<FEDChannel>& channels)
{
  static bool first = true;
  std::vector<SiStripCluster> out;

  auto const & det = clusterizer.stripByStripBegin(idet);
  state.reset(det);

  for (auto const& chan : channels) {
    auto perStripAdder = StripByStripAdder(clusterizer, state, out);
    unpackZS(chan, chan.iPair()*256, perStripAdder, idet);
    testUnpackZS(chan, chan.iPair()*256);
  }
  clusterizer.stripByStripEnd(state, out);

  if (first) {
    first = false;
    std::cout << "Printing clusters for detid " << idet << std::endl;
    for (const auto& cluster : out) {
      std::cout << "Cluster " << cluster.firstStrip() << " adcs: ";
      for (const auto& ampl : cluster.amplitudes()) {
        std::cout << (int) ampl << " ";
      }
      std::cout << std::endl;
    }
  }

  return out;
}

int main()
{
  Clusterizer clusterizer;
  Clusterizer::State state;

  FEDSet feds(fillFeds());
  for (auto idet : clusterizer.allDetIds()) {
    auto it = feds.find(idet);
    if (it != feds.end()) {
      auto out = fillClusters(idet, clusterizer, state, it->second);
    }
  }
}
