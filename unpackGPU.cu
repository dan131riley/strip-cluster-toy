#ifdef USE_GPU
#include <iostream>
#include <cassert>

#include <cuda_runtime.h>
#include <cub/util_debug.cuh>

#include "unpackGPU.cuh"
#include "cuda_rt_call.h"
#include "cudaCheck.h"

struct ChanLocStruct {
  void Fill(const ChannelLocsGPU& c);

  __host__ __device__ size_t size() const { return size_; }

  __host__ __device__ const uint8_t* input(uint32_t index) const { return input_[index]; }
  __host__ __device__ size_t inoff(uint32_t index) const { return inoff_[index]; }
  __host__ __device__ size_t offset(uint32_t index) const { return offset_[index]; }
  __host__ __device__ uint16_t length(uint32_t index) const { return length_[index]; }
  __host__ __device__ fedId_t fedID(uint32_t index) const { return fedID_[index]; }
  __host__ __device__ fedCh_t fedCh(uint32_t index) const { return fedCh_[index]; }


  const uint8_t** input_; // input raw data for channel
  size_t* inoff_;         // offset in input raw data
  size_t* offset_;        // global offset in alldata
  uint16_t* length_;      // length of channel data
  fedId_t* fedID_;
  fedCh_t* fedCh_;
  size_t size_;
};

void ChanLocStruct::Fill(const ChannelLocsGPU& c)
{
  input_ = c.input();
  inoff_ = c.inoff();
  offset_ = c.offset();
  length_ = c.length();
  fedID_ = c.fedID();
  fedCh_ = c.fedCh();
  size_ = c.size();
  }

constexpr auto kStripsPerChannel = SiStripConditionsBase::kStripsPerChannel;

__global__
static void unpackChannels(const ChanLocStruct* chanlocs, const SiStripConditionsGPU* conditions,
                           uint8_t* alldata, detId_t* detId, stripId_t* stripId,
                           float* noise, float* gain, bool* bad)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;

  const auto chan = nthreads*bid + tid;
  if (chan < chanlocs->size()) {
    const auto fedid = chanlocs->fedID(chan);
    const auto fedch = chanlocs->fedCh(chan);
    const auto detid = conditions->detID(fedid, fedch);
    const auto ipoff = kStripsPerChannel*conditions->iPair(fedid, fedch);

    const auto data = chanlocs->input(chan);
    const auto len = chanlocs->length(chan);

    if (data != nullptr && len > 0) {
      auto aoff = chanlocs->offset(chan);
      auto choff = chanlocs->inoff(chan);
      const auto end = aoff + len;

      while (aoff < end) {
        stripId[aoff] = invStrip;
        detId[aoff] = detid;
        alldata[aoff] = data[(choff++)^7];
        auto stripIndex = alldata[aoff++] + ipoff;
 
        stripId[aoff] = invStrip;
        detId[aoff] = detid;
        alldata[aoff] = data[(choff++)^7];
        const auto groupLength = alldata[aoff++];

        for (auto i = 0; i < groupLength; ++i) {
          noise[aoff] = conditions->noise(fedid, fedch, stripIndex);
          gain[aoff]  = conditions->gain(fedid, fedch, stripIndex);
          bad[aoff]   = conditions->bad(fedid, fedch, stripIndex);
          detId[aoff] = detid;
          stripId[aoff] = stripIndex++;
          alldata[aoff++] = data[(choff++)^7];
        }
      }
    }
  }
}

StripDataGPU::StripDataGPU(size_t size, cudaStream_t stream)
{
  alldataGPU_ = cudautils::make_device_unique<uint8_t[]>(size, stream);
  detIdGPU_ = cudautils::make_device_unique<detId_t[]>(size, stream);
  stripIdGPU_ = cudautils::make_device_unique<stripId_t[]>(size, stream);
  noiseGPU_ = cudautils::make_device_unique<float[]>(size, stream);
  gainGPU_ = cudautils::make_device_unique<float[]>(size, stream);
  badGPU_ = cudautils::make_device_unique<bool[]>(size, stream);
}

void unpackChannelsGPU(const ChannelLocsGPU& chanlocs, const SiStripConditionsGPU* conditions, StripDataGPU& stripdata, cudaStream_t stream)
{
  constexpr int nthreads = 128;
  const auto channels = chanlocs.size();
  const auto nblocks = (channels + nthreads - 1)/nthreads;

  ChanLocStruct chanstruct;
  chanstruct.Fill(chanlocs);
  auto chanstructGPU = cudautils::make_device_unique<ChanLocStruct>(stream);
  cudaCheck(cudaMemcpyAsync(chanstructGPU.get(), &chanstruct, sizeof(ChanLocStruct), cudaMemcpyDefault));
  
  unpackChannels<<<nblocks, nthreads, 0, stream>>>(chanstructGPU.get(), conditions,
                                                   stripdata.alldataGPU_.get(),
                                                   stripdata.detIdGPU_.get(),
                                                   stripdata.stripIdGPU_.get(),
                                                   stripdata.noiseGPU_.get(),
                                                   stripdata.gainGPU_.get(),
                                                   stripdata.badGPU_.get());
}

#endif
