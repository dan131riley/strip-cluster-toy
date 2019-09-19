#pragma once

#include <limits>
#include <cstdint>
#include <string>

using detId_t = uint32_t;
using fedId_t = uint16_t;
using fedCh_t = uint8_t;

class ChannelConditions {
public:
  static constexpr int kStripsPerChannel = 256;
  static constexpr detId_t invDet = std::numeric_limits<detId_t>::max();
  
  ChannelConditions(detId_t det, uint16_t pair, const float* noise, const float* gain, const bool* bad)
    : detId_(det), ipair_(pair), noise_(noise), gain_(gain), bad_(bad) {}

  detId_t detID() const { return detId_; }
  uint16_t iPair() const { return ipair_; }
  float noise(int strip) const { return noise_[strip-ipair_*kStripsPerChannel]; }
  float gain(int strip) const { return gain_[strip-ipair_*kStripsPerChannel]; }
  bool bad(int strip) const { return bad_[strip-ipair_*kStripsPerChannel]; }

private:
  detId_t detId_ = invDet;
  uint16_t ipair_;
  const float* noise_;
  const float* gain_;
  const bool* bad_;
};


class SiStripConditions {
public:
  static constexpr int kStripsPerChannel = ChannelConditions::kStripsPerChannel;
  static constexpr int kFedFirst = 50;
  static constexpr int kFedLast = 430;
  static constexpr int kFedCount = kFedLast - kFedFirst + 1;
  static constexpr int kChannelCount = 96;

  SiStripConditions(const std::string& file);
  SiStripConditions() {}

  const ChannelConditions operator()(fedId_t fed, fedCh_t channel) const;

private:
  float noise_[kFedCount][kChannelCount*kStripsPerChannel];
  float gain_[kFedCount][kChannelCount*kStripsPerChannel];
  bool bad_[kFedCount][kChannelCount*kStripsPerChannel];
  detId_t detID_[kFedCount][kChannelCount];
  uint16_t iPair_[kFedCount][kChannelCount];
};
