#ifdef USE_GPU
#include "unpackGPU.cuh"
#include "cuda_rt_call.h"

#include <cuda_runtime.h>
#include <cub/util_debug.cuh>

#include <iostream>

__global__
static void unpackChannels(const ChannelLocsBase* chanlocs, const SiStripConditionsGPU* conditions,
                           uint8_t* alldata, detId_t* detId, stripId_t* stripId)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;

  const auto chan = nthreads*bid + tid;
  if (chan < chanlocs->size()) {
    auto fedid = chanlocs->fedID(chan);
    auto fedch = chanlocs->fedCh(chan);
    auto detid = conditions->detID(fedid, fedch);

    const auto data = chanlocs->input(chan);

    if (data != nullptr) {
      auto aoff = chanlocs->offset(chan);
      auto choff = chanlocs->inoff(chan);

      for (auto k = 0; k < chanlocs->length(chan); ++k) {
        alldata[aoff] = data[choff^7];
        detId[aoff] = detid;
        
        if (chan == 0 && k < 8) {
          printf("Offset %lu/%lu data 0x%02x/0x%02x\n", choff^7, aoff, (int) data[choff^7], (int) alldata[aoff]);
        }
        aoff++; choff++;
      }
    }
  }
}

void unpackChannelsGPU(const ChannelLocsGPU& chanlocs, const SiStripConditionsGPU* conditions,
                       uint8_t* alldataGPU, detId_t* detIdGPU, stripId_t* stripIdGPU)
{
  constexpr int nthreads = 128;
  const auto channels = chanlocs.size();
  const auto nblocks = (channels + nthreads - 1)/nthreads;
  
  cudaDeviceSynchronize();
  unpackChannels<<<nblocks, nthreads>>>(chanlocs.onGPU(), conditions, alldataGPU, detIdGPU, stripIdGPU);
  cudaDeviceSynchronize();
  cudaError_t e = cudaGetLastError();
  CubDebugExit(e);
  std::cout << std::endl;
}

#endif
