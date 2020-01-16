#pragma once
#include "SiStripFEDBuffer.h"
#include "FEDChannel.h"

void unpackChannelsGPU(const ChannelLocsGPU& chanlocs, const SiStripConditionsGPU* conditions,
                       uint8_t* alldata, detId_t* detId, stripId_t* stripId,
                       float* noise, float* gain, bool* bad);
