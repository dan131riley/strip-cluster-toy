#include "SiStripConditions.h"

#include <fstream>

struct FileChannel {
  uint16_t fedId_;
  uint32_t  detId_;
  uint8_t fedCh_;
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
    auto& cc = channelAt(fc.fedId_, fc.fedCh_);
    cc.setDet(fc.detId_, fc.ipair_);
    for (auto i = 0; i < ChannelConditions::kStripsPerChannel; ++i) {
      cc.setStrip(i, fc.noise_[i], fc.gain_[i], fc.bad_[i]);
    }
  }
}