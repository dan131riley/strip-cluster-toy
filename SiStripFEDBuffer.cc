#include "SiStripFEDBuffer.h"


FEDBuffer::FEDBuffer(const uint8_t* fedBuffer, const uint16_t fedBufferSize, const bool allowBadBuffer)
  : orderedBuffer_(fedBuffer),bufferSize_(fedBufferSize)
{
  channels_.reserve(FEDCH_PER_FED);

  //build the correct type of FE header object
  feHeader_ = std::make_unique<FEDFullDebugHeader>(getPointerToDataAfterTrackerSpecialHeader());
  payloadPointer_ = getPointerToDataAfterTrackerSpecialHeader()+feHeader_->lengthInBytes();
  payloadLength_ = getPointerToByteAfterEndOfPayload()-payloadPointer_;

  //try to find channels
  validChannels_ = 0;
  findChannels();
}

void FEDBuffer::findChannels()
{
  uint16_t offsetBeginningOfChannel = 0;
  for (uint16_t i = 0; i < FEDCH_PER_FED; i++) {
    channels_.push_back(FEDChannel(payloadPointer_,offsetBeginningOfChannel));
    //get length and check that whole channel fits into buffer
    uint16_t channelLength = channels_.back().length();

    validChannels_++;
    const uint16_t offsetEndOfChannel = offsetBeginningOfChannel+channelLength;
    //add padding if necessary and calculate offset for begining of next channel
    if (!( (i+1) % FEDCH_PER_FEUNIT )) {
      uint8_t numPaddingBytes = 8 - (offsetEndOfChannel % 8);
      if (numPaddingBytes == 8) numPaddingBytes = 0;
      offsetBeginningOfChannel = offsetEndOfChannel + numPaddingBytes;
    } else {
      offsetBeginningOfChannel = offsetEndOfChannel;
    }
  }
}
