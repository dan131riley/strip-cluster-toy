#pragma once

#include "FEDChannel.h"

class FEDZSChannelUnpacker
{
public:
  static FEDZSChannelUnpacker zeroSuppressedModeUnpacker(const FEDChannel& channel);
  FEDZSChannelUnpacker();
  uint8_t sampleNumber() const;
  uint8_t adc() const;
  bool hasData(uint16_t extra = 0) const;
  FEDZSChannelUnpacker& operator ++ ();
  FEDZSChannelUnpacker& operator ++ (int);
private:
  //pointer to beginning of FED or FE data, offset of start of channel payload in data and length of channel payload
  FEDZSChannelUnpacker(const uint8_t* payload, const uint16_t channelPayloadOffset, const int16_t channelPayloadLength);
  void readNewClusterInfo();
  const uint8_t* data_;
  uint16_t currentOffset_;
  uint8_t currentStrip_;
  uint8_t valuesLeftInCluster_;
  uint16_t channelPayloadOffset_;
  uint16_t channelPayloadLength_;
};

inline void FEDZSChannelUnpacker::readNewClusterInfo()
{
  currentStrip_ = data_[(currentOffset_++)^7];
  valuesLeftInCluster_ = data_[(currentOffset_++)^7]-1;
  //std::cout << "New group offset " << currentOffset_ << " length " << (int) valuesLeftInCluster_+1 << " first strip " << (int) currentStrip_ << std::endl;
}


inline
FEDZSChannelUnpacker::FEDZSChannelUnpacker(const uint8_t* payload, const uint16_t channelPayloadOffset, const int16_t channelPayloadLength)
: data_(payload),
  currentOffset_(channelPayloadOffset),
  currentStrip_(0),
  valuesLeftInCluster_(0),
  channelPayloadOffset_(channelPayloadOffset),
  channelPayloadLength_(channelPayloadLength)
{
  if (channelPayloadLength_) readNewClusterInfo();
}

inline FEDZSChannelUnpacker FEDZSChannelUnpacker::zeroSuppressedModeUnpacker(const FEDChannel& channel)
{
  uint16_t length = channel.length();
  FEDZSChannelUnpacker result(channel.data(),channel.offset()+7,length-7);
  return result;
}

inline uint8_t FEDZSChannelUnpacker::sampleNumber() const
{
  return currentStrip_;
}

inline uint8_t FEDZSChannelUnpacker::adc() const
{
  return data_[currentOffset_^7];
}

inline bool FEDZSChannelUnpacker::hasData(uint16_t extra) const
{
  return (currentOffset_+extra < channelPayloadOffset_+channelPayloadLength_);
}

inline FEDZSChannelUnpacker& FEDZSChannelUnpacker::operator ++ ()
{
  if (valuesLeftInCluster_) {
    currentStrip_++;
    currentOffset_++;
    valuesLeftInCluster_--;
  } else {
    currentOffset_++;
    if (hasData(2)) {
      const uint8_t oldStrip = currentStrip_;
      readNewClusterInfo();
    }
  }
  return (*this);
}

inline FEDZSChannelUnpacker& FEDZSChannelUnpacker::operator ++ (int)
{
  ++(*this); return *this;
}
