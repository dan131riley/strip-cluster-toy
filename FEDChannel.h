#pragma once

#include <memory>

#include "SiStripConditions.h"
#include "cudaCompat.h"

class ChannelLocsGPU;

class ChannelLocsBase {
public:
  ChannelLocsBase(size_t size) : size_(size) {}
  ~ChannelLocsBase() {}

  void setChannelLoc(uint32_t index, const uint8_t* input, size_t inoff, size_t offset, uint16_t length, fedId_t fedID, fedCh_t fedCh);

  __host__ __device__ size_t size() const { return size_; }

  __host__ __device__ const uint8_t* input(uint32_t index) const { return input_[index]; }
  __host__ __device__ size_t inoff(uint32_t index) const { return inoff_[index]; }
  __host__ __device__ size_t offset(uint32_t index) const { return offset_[index]; }
  __host__ __device__ uint16_t length(uint32_t index) const { return length_[index]; }
  __host__ __device__ fedId_t fedID(uint32_t index) const { return fedID_[index]; }
  __host__ __device__ fedCh_t fedCh(uint32_t index) const { return fedCh_[index]; }

  const uint8_t** input() const { return input_; }
  size_t* inoff() const { return inoff_; }
  size_t* offset() const { return offset_; }
  uint16_t* length() const { return length_; }
  fedId_t* fedID() const { return fedID_; }
  fedCh_t* fedCh() const { return fedCh_; }

protected:
  const uint8_t** input_  = nullptr; // input raw data for channel
  size_t*   inoff_        = nullptr; // offset in input raw data
  size_t*   offset_       = nullptr; // global offset in alldata
  uint16_t* length_       = nullptr; // length of channel data
  fedId_t*  fedID_        = nullptr;
  fedCh_t*  fedCh_        = nullptr;
  size_t    size_         = 0;
};

class ChannelLocs : public ChannelLocsBase {
public:
  ChannelLocs(size_t size);
  ~ChannelLocs();
};

class ChannelLocsGPU : public ChannelLocsBase {
public:
  ChannelLocsGPU(size_t size);
  ~ChannelLocsGPU();
  const ChannelLocsBase* onGPU() const { return onGPU_; }
  void reset(const ChannelLocsBase&, const std::vector<uint8_t*>& inputGPU);
private:
  ChannelLocsBase* onGPU_ = nullptr;
};

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
