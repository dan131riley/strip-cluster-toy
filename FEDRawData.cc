/** \file
   implementation of class FedRawData

   \author Stefano ARGIRO
   \date 28 Jun 2005
*/

#include <cassert>

#include "FEDRawData.h"

using namespace std;

FEDRawData::FEDRawData(size_t size, cudaStream_t stream)
  : size_(size) {
  assert(size_ % 8 == 0);
  data_ = cudautils::make_host_unique<unsigned char[]>(size_, stream);
}

FEDRawData::~FEDRawData() {}
