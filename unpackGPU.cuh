#pragma once
#include "FEDChannel.h"

void unpackChannelsGPU(const ChannelLocsGPU& chanlocs, const SiStripConditionsGPU* conditions,
                       uint8_t* alldataGPU, detId_t* detIdGPU, stripId_t* stripIdGPU);
