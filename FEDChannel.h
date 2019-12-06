#pragma once

#include <memory>

#include "Clusterizer.h"

//holds information about position of a channel in the buffer for use by unpacker
class FEDChannel {
public:
  FEDChannel(const uint8_t* const data, const size_t offset, const uint16_t length);
  //gets length from first 2 bytes (assuming normal FED channel)
  FEDChannel(const uint8_t* const data, const size_t offset);
  uint16_t length() const { return length_; }
  const uint8_t* data() const { return data_; }
  size_t offset() const { return offset_; }
  uint16_t cmMedian(const uint8_t apvIndex) const;
  //third byte of channel data for normal FED channels
  uint8_t packetCode() const;

private:
  friend class FEDBuffer;
  const uint8_t* data_;
  size_t offset_;
  uint16_t length_;
};

inline FEDChannel::FEDChannel(const uint8_t* const data, const size_t offset) : data_(data), offset_(offset) {
  length_ = (data_[(offset_) ^ 7] + (data_[(offset_ + 1) ^ 7] << 8));
}

inline FEDChannel::FEDChannel(const uint8_t*const data, const size_t offset, const uint16_t length)
  : data_(data),
    offset_(offset),
    length_(length)
{
}
