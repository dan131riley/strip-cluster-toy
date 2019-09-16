#pragma once

#include <cstdint>
#include <cstddef>

#include "FEDChannel.h"

static const uint16_t FEDCH_PER_FEUNIT = 12;
static const uint16_t FEUNITS_PER_FED = 8;
static const uint16_t FEDCH_PER_FED = FEDCH_PER_FEUNIT * FEUNITS_PER_FED;  // 96

class FEDFullDebugHeader
{
public:
  static const size_t FULL_DEBUG_HEADER_SIZE_IN_64BIT_WORDS = FEUNITS_PER_FED*2;
  static const size_t FULL_DEBUG_HEADER_SIZE_IN_BYTES = FULL_DEBUG_HEADER_SIZE_IN_64BIT_WORDS*8;
  size_t lengthInBytes() const { return FULL_DEBUG_HEADER_SIZE_IN_BYTES; }
  explicit FEDFullDebugHeader(const uint8_t* headerBuffer) {}
};


class FEDBuffer
{
public:
  //construct from buffer
  FEDBuffer(const uint8_t* fedBuffer, const uint16_t fedBufferSize, const bool allowBadBuffer = false);
  ~FEDBuffer() {}

  const uint8_t* getPointerToDataAfterTrackerSpecialHeader() const { return orderedBuffer_ + 16; }
  const uint8_t* getPointerToByteAfterEndOfPayload() const { return orderedBuffer_+bufferSize_-8; }

  const FEDChannel& channel(const uint8_t internalFEDChannelNum) const { return channels_[internalFEDChannelNum]; }
  const std::vector<FEDChannel>& channels() const { return channels_; }
  uint8_t validChannels() const { return validChannels_; }

private:
  void findChannels();
  std::unique_ptr<FEDFullDebugHeader> feHeader_;
  const uint8_t* payloadPointer_;
  uint16_t payloadLength_;
  std::vector<FEDChannel> channels_;
  const uint8_t* orderedBuffer_;
  const size_t bufferSize_;
  uint8_t validChannels_;
};
