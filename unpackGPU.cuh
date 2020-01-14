#pragma once
#include "SiStripFEDBuffer.h"
#include "FEDChannel.h"

void unpackChannelsGPU(const ChannelLocsGPU& chanlocs, const SiStripConditionsGPU* conditions,
                       uint8_t* alldataGPU, detId_t* detIdGPU, stripId_t* stripIdGPU,
                       fedId_t* fedId, fedCh_t* fedCh);
