#include <iostream>
#include <chrono>
#include <cmath>
#include <cassert>

#include "tiny_dnn/tiny_dnn.h"

#include "util.cpp"

// allocates and returns a string. Allocation must be freed by the
// caller
char *predict(char *model_path, char *locdom, char *whole) {


  tiny_dnn::vec_t input_vec;
  std::cout << "Model path " << model_path << std::endl;
  std::cout << "locdom " << locdom << std::endl;
  std::cout << "whole  " << whole << std::endl;

  append_to_vec_from_ssv(input_vec, std::string(locdom));
  append_to_vec_from_ssv(input_vec, std::string(whole));

  tiny_dnn::vec_t whole_vec = vec_from_ssv(std::string(whole));

#ifdef MEASURE
  std::cout << "Testing load" << std::endl;
  auto start = get_time();
#endif

  tiny_dnn::network<tiny_dnn::sequential> nn;
  nn.load(std::string(model_path)+std::string("modelbest"));

#ifdef MEASURE
  auto end = get_time();
  std::cout << "Loading time " << diff_time(end, start)  << 
               " microsecondss.\n";

  std::cout << "Testing inference" << std::endl;
  start = get_time();
#endif

  tiny_dnn::vec_t prediction = nn.predict(input_vec);

#ifdef MEASURE
  end = get_time();
  std::cout << "Loading time " << diff_time(end, start) <<
               " microsecondss.\n";
#endif

  char *res = (char *)calloc(100, sizeof(char));
  tiny_dnn::vec_t norm_pred = normalize_prediction(prediction, whole_vec);

  for(auto &val: norm_pred) {
    strcat(res, std::to_string((int)val).c_str());
    strcat(res, " ");
  }
  return res;
}

// just to test
// in the end I'd like this to be a library that can be linked together
// with the application
// arguments : ./predict <model_path> <locdom> <whole>
int main(int argc, char *argv[]) {
  std::cout << "Application started\n";
  assert(argc==4);
  std::cout << "Calling predict\n";

  char *res = predict(argv[1], argv[2], argv[3]);
  std::cout << "Parsing prediction\n";
  char *tok = strtok(res, " ");
  std::cout << "Writing to file\n";
  FILE *pred_file = fopen("prediction", "w");
  do {
    fprintf(pred_file, "%s\n", tok);
  } while((tok = strtok (NULL, " ")));
  fclose(pred_file);
}
