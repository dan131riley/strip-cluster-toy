/** \file
   implementation of class FedRawData

   \author Stefano ARGIRO
   \date 28 Jun 2005
*/

#include <cassert>

#include "FEDRawData.h"

using namespace std;

FEDRawData::FEDRawData() {}

FEDRawData::FEDRawData(size_t newsize) : data_(newsize) {
  assert(newsize % 8 == 0);
}

FEDRawData::FEDRawData(const FEDRawData &in) : data_(in.data_) {}
FEDRawData::~FEDRawData() {}
const unsigned char *FEDRawData::data() const { return data_.data(); }

unsigned char *FEDRawData::data() { return data_.data(); }

void FEDRawData::resize(size_t newsize) {
  if (size() == newsize)
    return;

  data_.resize(newsize);
  assert(newsize % 8 == 0);
}
