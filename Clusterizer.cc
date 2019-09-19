#include <numeric>
#include <cmath>
#include <iostream>

#include "Clusterizer.h"

template<typename Iter>
inline float chargePerCM(Iter a, Iter b) {
  return float(std::accumulate(a,b,int(0)))/0.047f;
}

inline bool
Clusterizer::candidateEnded(State const & state, const uint16_t& testStrip) const {
  uint16_t holes = testStrip - state.lastStrip - 1;
  return ( ( (!state.ADCs.empty())  &&                    // a candidate exists, and
	     (holes > MaxSequentialHoles )       // too many holes if not all are bad strips, and
	     ) && 
	   ( holes > MaxSequentialBad ||       // (too many bad strips anyway, or 
	     !state.det().allBadBetween( state.lastStrip, testStrip ) // not all holes are bad strips)
	     )
	   );
}

inline void
Clusterizer::addToCandidate(State & state, uint16_t strip, uint8_t adc) const { 
  float Noise = state.det().noise( strip );
  if(  adc < static_cast<uint8_t>( Noise * ChannelThreshold) || state.det().bad(strip) )
    return;

  if(state.candidateLacksSeed) state.candidateLacksSeed  =  adc < static_cast<uint8_t>( Noise * SeedThreshold);
  if(state.ADCs.empty()) state.lastStrip = strip - 1; // begin candidate
  while( ++state.lastStrip < strip ) state.ADCs.push_back(0); // pad holes

  state.ADCs.push_back( adc );
  state.noiseSquared += Noise*Noise;
}

template <class T> inline void
Clusterizer::endCandidate(State & state, T& out) const {
  if(candidateAccepted(state)) {
    applyGains(state);
    appendBadNeighbors(state);
    if(chargePerCM(state.ADCs.begin(), state.ADCs.end()) > minGoodCharge)
      out.push_back(SiStripCluster(firstStrip(state), state.ADCs.begin(), state.ADCs.end()));
  }
  clearCandidate(state);  
}

inline bool
Clusterizer::candidateAccepted(State const & state) const {
  return ( !state.candidateLacksSeed &&
	   state.noiseSquared * ClusterThresholdSquared
	   <=  std::pow( float(std::accumulate(state.ADCs.begin(),state.ADCs.end(), int(0))), 2.f));
}

inline void
Clusterizer::applyGains(State & state) const {
  uint16_t strip = firstStrip(state);
  for( auto &  adc :  state.ADCs) {
    auto charge = int( float(adc)/state.det().gain(strip++) + 0.5f ); //adding 0.5 turns truncation into rounding
    if(adc < 254) adc = ( charge > 1022 ? 255 : 
			  ( charge >  253 ? 254 : charge ));
  }
}

inline void
Clusterizer::appendBadNeighbors(State & state) const {
  uint8_t max = MaxAdjacentBad;
  while(0 < max--) {
    if( state.det().bad( firstStrip(state)-1) ) { state.ADCs.insert( state.ADCs.begin(), 0);  }
    if( state.det().bad(  state.lastStrip + 1) ) { state.ADCs.push_back(0); state.lastStrip++; }
  }
}

Clusterizer::Det
Clusterizer::stripByStripBegin(fedId_t fedId) const {
  return Det(conditions_, fedId);
}

void
Clusterizer::stripByStripAdd(State & state, uint16_t strip, uint8_t adc, std::vector<SiStripCluster>& out) const {
  if(candidateEnded(state,strip)) endCandidate(state,out);
  addToCandidate(state, SiStripDigi(strip,adc));
}


void
Clusterizer::stripByStripEnd(State & state, std::vector<SiStripCluster>& out) const { 
  endCandidate(state, out);
}
