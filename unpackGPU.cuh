#pragma once
#include "SiStripFEDBuffer.h"
#include "FEDChannel.h"
#include "device_unique_ptr.h"

class StripDataGPU {
public:
  StripDataGPU(size_t size, cudaStream_t stream);

  cudautils::device::unique_ptr<uint8_t[]> alldataGPU_;
  cudautils::device::unique_ptr<detId_t[]> detIdGPU_;
  cudautils::device::unique_ptr<stripId_t[]> stripIdGPU_;
  cudautils::device::unique_ptr<float[]> noiseGPU_;
  cudautils::device::unique_ptr<float[]> gainGPU_;
  cudautils::device::unique_ptr<bool[]> badGPU_;
};

void unpackChannelsGPU(const ChannelLocsGPU& chanlocs, const SiStripConditionsGPU* conditions, StripDataGPU& stripdata, cudaStream_t stream);
