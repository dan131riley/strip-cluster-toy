#include "SiStripConditions.h"

#include <fstream>

struct FileChannel {
  fedId_t fedId_;
  detId_t  detId_;
  fedCh_t fedCh_;
  uint16_t ipair_;
  float noise_[ChannelConditions::kStripsPerChannel];
  float gain_[ChannelConditions::kStripsPerChannel];
  bool bad_[ChannelConditions::kStripsPerChannel];
};

SiStripConditions::SiStripConditions(const std::string& file)
{
  std::ifstream datafile(file, std::ios::in | std::ios::binary);

  FileChannel fc;

  while (datafile.read((char*) &fc, sizeof(fc)).gcount() == sizeof(fc)) {
    auto fedi = fc.fedId_ - kFedFirst;
    auto fch = fc.fedCh_;
    detID_[fedi][fch] = fc.detId_;
    iPair_[fedi][fch] = fc.ipair_;
    auto choff = fch*kStripsPerChannel;
    for (auto i = 0; i < ChannelConditions::kStripsPerChannel; ++i, ++choff) {
      noise_[fedi][choff] = fc.noise_[i];
      gain_[fedi][choff] = fc.gain_[i];
      bad_[fedi][choff] = fc.bad_[i];
    }
  }
}

const ChannelConditions SiStripConditions::operator()(fedId_t fed, fedCh_t channel) const
{
  fed -= kFedFirst;
  const auto detID = detID_[fed][channel];
  const auto pair = iPair_[fed][channel];
  const auto choff = channel*kStripsPerChannel;
  return ChannelConditions(detID, pair, &noise_[fed][choff], &gain_[fed][choff], &bad_[fed][choff]);
}
