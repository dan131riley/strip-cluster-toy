#pragma once

#include <vector>
#include <map>
#include <fstream>

#include "SiStripConditions.h"
#include "Strip.h"

class Clusterizer {
public:
  Clusterizer(const SiStripConditions* conditions) : conditions_(conditions) {}

  // state of detID
  class Det {
  public:
    Det(const SiStripConditions* conditions, fedId_t fedID)
      : conditions_(conditions), fedID_(fedID) {}
    float noise(const uint16_t& strip) const { return (*conditions_)(fedID_, fedCh_).noise(strip); }
    float gain(const uint16_t& strip)  const { return (*conditions_)(fedID_, fedCh_).gain(strip); }
    bool bad(const uint16_t& strip)    const { return (*conditions_)(fedID_, fedCh_).bad(strip); }
    bool allBadBetween(uint16_t L, const uint16_t& R) const { while( ++L < R  &&  bad(L)) {}; return L == R; }
    detId_t id() const { return (*conditions_)(fedID_, fedCh_).detID(); }
    void setFedCh(fedCh_t fedCh) { fedCh_ = fedCh; }
  private:
    const SiStripConditions* conditions_;
    fedId_t fedID_;
    fedCh_t fedCh_;
  };

  //state of the candidate cluster
  struct State {
    State(Det const & idet) : mp_det(&idet) { ADCs.reserve(128);}
    State() : mp_det(nullptr) { ADCs.reserve(128); }
    Det const& det() const { return *mp_det; }
    void reset(Det const& idet) {
      mp_det = &idet;
      ADCs.clear();
      lastStrip = 0; noiseSquared = 0; candidateLacksSeed = true;
    }
    std::vector<uint8_t> ADCs;  
    uint16_t lastStrip=0;
    float noiseSquared=0;
    bool candidateLacksSeed=true;
  private:
    Det const * mp_det;
  };

  Det stripByStripBegin(fedId_t fedId) const;

  void stripByStripEnd(State & state, std::vector<SiStripCluster>& out) const;
  void stripByStripAdd(State & state, uint16_t strip, uint8_t adc, std::vector<SiStripCluster>& out) const;

private:
  //constant methods with state information
  uint16_t firstStrip(State const & state) const {return state.lastStrip - state.ADCs.size() + 1;}
  bool candidateEnded(State const & state, const uint16_t&) const;
  bool candidateAccepted(State const & state) const;

  //state modification methods
  template<class T> void endCandidate(State & state, T&) const;
  void clearCandidate(State & state) const { state.candidateLacksSeed = true;  state.noiseSquared = 0;  state.ADCs.clear();}
  void addToCandidate(State & state, const SiStripDigi& digi) const { addToCandidate(state, digi.strip(),digi.adc());}
  void addToCandidate(State & state, uint16_t strip, uint8_t adc) const;
  void appendBadNeighbors(State & state) const;
  void applyGains(State & state) const;

  float ChannelThreshold = 2.0, SeedThreshold = 3.0, ClusterThresholdSquared = 25.0;
  uint8_t MaxSequentialHoles = 0, MaxSequentialBad = 1, MaxAdjacentBad = 0;
  float minGoodCharge = 1620.0;

  const SiStripConditions* conditions_;
};


