#include <iostream>
#include <chrono>
#include <cmath>
#include <cassert>

#include "tiny_dnn/tiny_dnn.h"

#include "util.cpp"

tiny_dnn::vec_t normalize_prediction(tiny_dnn::vec_t &pred,
                                     tiny_dnn::vec_t &whole) {
  assert(pred.size() == 4);

  tiny_dnn::vec_t ret;

  ret = { floor(pred[0]),
          ceil(pred[1]),
          floor(pred[2]),
          ceil(pred[3]) };

  ret = { ret[0] >= whole[0] ? ret[0] : (int)whole[0],
          ret[1] <= whole[1] ? ret[1] : (int)whole[1],
          ret[2] >= whole[2] ? ret[2] : (int)whole[2],
          ret[3] <= whole[3] ? ret[3] : (int)whole[3]};

  ret = { ret[0] <= whole[1] ? ret[0] : (int)whole[1],
          ret[1] >= whole[0] ? ret[1] : (int)whole[0],
          ret[2] <= whole[3] ? ret[2] : (int)whole[3],
          ret[3] >= whole[2] ? ret[3] : (int)whole[2]};

  return ret;
}

// allocates and returns a string. Allocation must be freed by the
// caller
const char *predict(const char *locdom, const char *whole) {


  tiny_dnn::vec_t input_vec;
  append_to_vec_from_ssv(input_vec, std::string(locdom));
  append_to_vec_from_ssv(input_vec, std::string(whole));

  tiny_dnn::vec_t whole_vec = vec_from_ssv(std::string(whole));

  std::cout << "Testing load" << std::endl;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

  tiny_dnn::network<tiny_dnn::sequential> nn;
  nn.load("test-model");

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Loading time " << 
    std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() <<
    " microsecondss.\n";


  std::cout << "Testing inference" << std::endl;
  start = std::chrono::steady_clock::now();

  tiny_dnn::vec_t prediction = nn.predict(input_vec);

  end = std::chrono::steady_clock::now();
  std::cout << "Loading time " <<
    std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()<<
    " microsecondss.\n";

  char *res = (char *)calloc(100, sizeof(char));
  tiny_dnn::vec_t norm_pred = normalize_prediction(prediction, whole_vec);

  for(auto &val: norm_pred) {
    strcat(res, std::to_string((int)val).c_str());
    strcat(res, " ");
  }
  return res;
}

// just to test
int main(int argc, char *argv[]) {

  const char *res = predict("0 31 0 31", "0 63 0 63");
  printf("%s\n", res);
}
