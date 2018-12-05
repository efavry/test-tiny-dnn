#include <iostream>
#include <cmath>
#include "tiny_dnn/tiny_dnn.h"

static inline std::string vec_repr(const tiny_dnn::vec_t &vec) {
  std::string ret = "[";
  for(size_t i = 0 ; i < vec.size() ; i++) {
    ret += std::to_string(static_cast<float>(vec[i]));

    //std::cout << static_cast<float>(vec[i]) << std::endl;
    ret += ",";
  }
  ret += "]";
  return ret;
}

static inline float find_overlap(const tiny_dnn::vec_t &target,
                                 const tiny_dnn::vec_t &pred) {
  auto ret = 1.0;
  for(size_t i = 0 ; i < target.size()-1 ; i+=2) {
    auto tlo = target[i], thi = target[i+1];
    auto plo = pred[i], phi = pred[i+1];

    auto overlap = std::min(thi, phi) - std::max(tlo, plo);
    overlap = overlap < 0 ? 0 : overlap;
    ret *= overlap;
  }

  //std::cout << "Overlap between " << vec_repr(target) << " and "
            //<< vec_repr(pred) << " is " << ret << std::endl;
  return ret;
}

static inline float rec_size(const tiny_dnn::vec_t &rec) {
  float size = 1.0;
  for(size_t i = 0 ; i < rec.size()-1 ; i+=2) {
    auto lo = rec[i], hi = rec[i+1];
    size *= (hi-lo);
  }
  return size;
}

static inline float perf_efficiency(const tiny_dnn::vec_t &target, 
                                    const tiny_dnn::vec_t &pred) {
  auto target_sz = rec_size(target);
  if(target_sz == 0) // if target is empty, we have perfect perf eff.
    return 1;
  auto overlap = find_overlap(target, pred);
  //std::cout << "Perf efficiency of target=" << vec_repr(target) <<
                                 //" pred=" << vec_repr(pred) <<
                                 //" is " << overlap/target_sz << std::endl;
  return overlap/target_sz;
}

static inline float mem_efficiency(const tiny_dnn::vec_t &target,
                                   const tiny_dnn::vec_t &pred) {
  auto pred_sz = rec_size(pred);
  auto target_sz = rec_size(target);

  if(pred_sz == 0) {
    if(target_sz == 0) {
      //We were supposed to predict empty domain and we did
      return 1;
    }
    else {
      // target size was not zero but we haven't brough anything that we are not
      // supposed to, so memory efficiency is still 1
      return 1;
    }
  }
 
  auto overlap = find_overlap(target, pred);
  return overlap/pred_sz;
}

