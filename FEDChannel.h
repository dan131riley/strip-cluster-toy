#pragma once

#include <memory>

#include "Clusterizer.h"

//holds information about position of a channel in the buffer for use by unpacker
class FEDChannel
{
public:
  //FEDChannel(const uint8_t*const data, const size_t offset, const uint16_t length, fedId_t feded, fedCh_t fedch);
  FEDChannel() {}
  FEDChannel(std::ifstream& file);
  uint16_t length() const { return length_; }
  const uint8_t* data() const { return data_.get(); }
  size_t offset() const { return offset_; }
  //third byte of channel data for normal FED channels
  uint8_t packetCode() const { return data_[(offset_+2)^7]; }
  fedId_t fedId() const { return fedid_; }
  fedCh_t fedCh() const { return fedch_; }
  uint16_t iPair() const { return ipair_; }
private:
  std::unique_ptr<uint8_t[]> data_;
  size_t offset_;
  uint16_t length_;
  uint16_t ipair_;
  fedId_t fedid_;
  fedCh_t fedch_;
};

using FEDSet = std::map<detId_t, std::vector<FEDChannel>>;
