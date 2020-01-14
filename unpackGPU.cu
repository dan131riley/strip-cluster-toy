#ifdef USE_GPU
#include "unpackGPU.cuh"
#include "cuda_rt_call.h"

#include <cuda_runtime.h>
#include <cub/util_debug.cuh>

#include <iostream>
#include <cassert>

constexpr auto kStripsPerChannel = SiStripConditionsBase::kStripsPerChannel;

__global__
static void unpackChannels(const ChannelLocsBase* chanlocs, const SiStripConditionsGPU* conditions,
                           uint8_t* alldata, detId_t* detId, stripId_t* stripId,
                           fedId_t* fedId, fedCh_t* fedCh)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;

  const auto chan = nthreads*bid + tid;
  if (chan < chanlocs->size()) {
    auto fedid = chanlocs->fedID(chan);
    auto fedch = chanlocs->fedCh(chan);
    auto detid = conditions->detID(fedid, fedch);
    auto ipair = conditions->iPair(fedid, fedch);

    const auto data = chanlocs->input(chan);
    const auto len = chanlocs->length(chan);

    if (data != nullptr && len > 0) {
      auto aoff = chanlocs->offset(chan);
      auto choff = chanlocs->inoff(chan);

      for (auto i = 0; i < len; ++i) {
        detId[aoff] = detid;
        fedId[aoff] = fedid;
        fedCh[aoff] = fedch;
        stripId[aoff] = invStrip;
        alldata[aoff++] = data[(choff++)^7];
      }

      aoff = chanlocs->offset(chan);
      const auto end = aoff + len;

      while (aoff < end) {
        auto stripIndex = alldata[aoff++] + kStripsPerChannel*ipair;
        const auto groupLength = alldata[aoff++];

        for (auto i = 0; i < groupLength; ++i) {
          stripId[aoff++] = stripIndex++;
        }
      }
      assert(aoff == end);
    }
  }
}

void unpackChannelsGPU(const ChannelLocsGPU& chanlocs, const SiStripConditionsGPU* conditions,
                       uint8_t* alldataGPU, detId_t* detIdGPU, stripId_t* stripIdGPU,
                       fedId_t* fedId, fedCh_t* fedCh)
{
  constexpr int nthreads = 128;
  const auto channels = chanlocs.size();
  const auto nblocks = (channels + nthreads - 1)/nthreads;
  
  unpackChannels<<<nblocks, nthreads>>>(chanlocs.onGPU(), conditions, alldataGPU, detIdGPU, stripIdGPU, fedId, fedCh);
}

#endif
