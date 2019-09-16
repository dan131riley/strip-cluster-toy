#pragma once

#include <limits>
#include <cstdint>
#include <string>

class ChannelConditions {
public:
  static constexpr int kStripsPerChannel = 256;
  static constexpr uint32_t invDet = std::numeric_limits<uint32_t>::max();
  
  ChannelConditions() {}
  ChannelConditions(uint32_t det, uint16_t pair)
    : detId_(det), ipair_(pair) {}
  void setStrip(uint16_t strip, float noise, float gain, bool bad) {
    noise_[strip] = noise;
    gain_[strip] = gain;
    bad_[strip] = bad;
  }
  void setDet(uint32_t detId, uint16_t iPair) { detId_ = detId; ipair_ = iPair; }

  uint32_t detID() const { return detId_; }
  uint16_t iPair() const { return ipair_; }
  float noise(int strip) const { return noise_[strip-ipair_*kStripsPerChannel]; }
  float gain(int strip) const { return gain_[strip-ipair_*kStripsPerChannel]; }
  bool bad(int strip) const { return bad_[strip-ipair_*kStripsPerChannel]; }

private:
  uint32_t  detId_ = invDet;
  uint16_t ipair_;
  float noise_[kStripsPerChannel];
  float gain_[kStripsPerChannel];
  bool bad_[kStripsPerChannel];
};


class SiStripConditions {
public:
  static constexpr int kFedFirst = 50;
  static constexpr int kFedLast = 430;
  static constexpr int kFedCount = kFedLast - kFedFirst + 1;
  static constexpr int kChannelCount = 96;

  SiStripConditions(const std::string& file);
  SiStripConditions() {}

  const ChannelConditions& operator()(uint16_t fed, uint8_t channel) const { return channels_[fed - kFedFirst][channel]; }

private:
  ChannelConditions& channelAt(uint16_t fed, uint8_t channel) { return channels_[fed - kFedFirst][channel]; }
  ChannelConditions channels_[kFedCount][kChannelCount];
};
