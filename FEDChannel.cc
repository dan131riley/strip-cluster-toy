#include <cassert>

#ifdef USE_GPU
#include <cuda_runtime.h>
#include "cuda_rt_call.h"
#endif

#include "FEDChannel.h"

void ChannelLocsBase::setChannelLoc(uint32_t index, const uint8_t* input, size_t inoff, size_t offset,
                                    uint16_t length, fedId_t fedID, fedCh_t fedCh)
{
  input_[index] = input;
  inoff_[index] = inoff;
  offset_[index] = offset;
  length_[index] = length;
  fedID_[index] = fedID;
  fedCh_[index] = fedCh;
}

ChannelLocs::ChannelLocs(size_t size)
  : ChannelLocsBase(size)
{
  if (size > 0) {
    input_ = new const uint8_t*[size];
    inoff_ = new size_t[size];
    offset_ = new size_t[size];
    length_ = new uint16_t[size];
    fedID_ = new fedId_t[size];
    fedCh_ = new fedCh_t[size];
  }
}

ChannelLocs::~ChannelLocs()
{
  if (size() > 0) {
    delete [] input_;
    delete [] inoff_;
    delete [] offset_;
    delete [] length_;
    delete [] fedID_;
    delete [] fedCh_;
  }
}

ChannelLocsGPU::ChannelLocsGPU(size_t size)
  : ChannelLocsBase(size)
{
#ifdef USE_GPU
  if (size > 0) {
    CUDA_RT_CALL(cudaMalloc((void**) &input_, sizeof(uint8_t*)*size_));
    CUDA_RT_CALL(cudaMalloc((void**) &inoff_, sizeof(size_t)*size_));
    CUDA_RT_CALL(cudaMalloc((void**) &offset_, sizeof(size_t)*size_));
    CUDA_RT_CALL(cudaMalloc((void**) &length_, sizeof(uint16_t)*size_));
    CUDA_RT_CALL(cudaMalloc((void**) &fedID_, sizeof(fedId_t)*size_));
    CUDA_RT_CALL(cudaMalloc((void**) &fedCh_, sizeof(fedCh_t)*size_));
    CUDA_RT_CALL(cudaMalloc((void**) &onGPU_, sizeof(ChannelLocsBase)));
    CUDA_RT_CALL(cudaMemcpyAsync(onGPU_, this, sizeof(ChannelLocsBase), cudaMemcpyDefault));
  }
#endif
}

void ChannelLocsGPU::reset(const ChannelLocsBase& c, const std::vector<uint8_t*>& inputGPU)
{
#ifdef USE_GPU
  assert(c.size() == size_);
  CUDA_RT_CALL(cudaMemcpyAsync(input_, inputGPU.data(), sizeof(uint8_t*)*size_, cudaMemcpyDefault));
  CUDA_RT_CALL(cudaMemcpyAsync(inoff_, c.inoff(), sizeof(size_t)*size_, cudaMemcpyDefault));
  CUDA_RT_CALL(cudaMemcpyAsync(offset_, c.offset(), sizeof(size_t)*size_, cudaMemcpyDefault));
  CUDA_RT_CALL(cudaMemcpyAsync(length_, c.length(), sizeof(uint16_t)*size_, cudaMemcpyDefault));
  CUDA_RT_CALL(cudaMemcpyAsync(fedID_, c.fedID(), sizeof(fedId_t)*size_, cudaMemcpyDefault));
  CUDA_RT_CALL(cudaMemcpyAsync(fedCh_, c.fedCh(), sizeof(fedCh_t)*size_, cudaMemcpyDefault));
#endif
}

ChannelLocsGPU::~ChannelLocsGPU()
{
#ifdef USE_GPU
  if (size() > 0) {
    cudaFree(input_);
    cudaFree(inoff_);
    cudaFree(offset_);
    cudaFree(length_);
    cudaFree(fedID_);
    cudaFree(fedCh_);
    cudaFree(onGPU_);
  }
#endif
}
