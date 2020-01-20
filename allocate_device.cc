#include <limits>

#include "cudaCheck.h"
#include "allocate_device.h"
#include "getCachingDeviceAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cudautils::allocator::binGrowth, cudautils::allocator::maxBin);
}

namespace cudautils {
  void *allocate_device(int dev, size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if (nbytes > maxAllocationSize) {
      throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                               " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
    }
    cudaCheck(cudautils::allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, stream));
    return ptr;
  }

  void free_device(int device, void *ptr) {
    cudaCheck(cudautils::allocator::getCachingDeviceAllocator().DeviceFree(device, ptr));
  }

}  // namespace cudautils
