/*
 * Recopied from tiny dnn here is the original Copyright
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>
#include <chrono>

#include "tiny_dnn/tiny_dnn.h"

#include "util.cpp"

//#define CNN_TASK_SIZE 32

static void train(std::istream &data_stream,
                  double learning_rate,
                  const int n_train_epochs,
                  const int n_minibatch,
                  tiny_dnn::core::backend_t backend_type)
{
  // specify loss-function and learning strategy
  tiny_dnn::network<tiny_dnn::sequential> nn;
  tiny_dnn::adam optimizer;

  construct_net(nn, backend_type);

#ifdef DEBUG
  std::cerr << "model constructed..." << std::endl;
#endif

  tiny_dnn::timer t;

  optimizer.alpha *=
    std::min(tiny_dnn::float_t(4),
             static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

  int epoch = 1;

  // create dummy callbacks
  auto on_enumerate_epoch = [&]() { };
  auto on_enumerate_minibatch = [&]() { };

  // training //train_once exists !
  std::string line;
  std::vector<tiny_dnn::vec_t> data_vec;
  std::vector<tiny_dnn::vec_t> label_vec;
  
  std::getline(data_stream, line);
  while(line != "EXIT") {
    if(line == "BATCH") {
#ifdef DEBUG
      for(auto &datapoint: data_vec) {
        for(auto &data: datapoint) {
          std::cout << data << " ";
        }
        std::cout << std::endl;
      }
#endif
      std::cout << "Fit called with " << data_vec.size() <<
        " data point scanned\n";

      nn.fit<tiny_dnn::mse>(optimizer,
                            data_vec,
                            label_vec,
                            n_minibatch,
                            n_train_epochs,
                            on_enumerate_minibatch,
                            on_enumerate_epoch);
      data_vec.clear();
      label_vec.clear();
    }
    else {
      parse_and_append(line, data_vec, label_vec);
    }
    std::getline(data_stream, line);
  }
#ifdef DEBUG
      for(auto &datapoint: data_vec) {
        for(auto &data: datapoint) {
          std::cout << data << " ";
        }
        std::cout << std::endl;
      }
#endif
  nn.fit<tiny_dnn::mse>(optimizer,
                          data_vec,
                          label_vec,
                          n_minibatch,
                          n_train_epochs,
                          on_enumerate_minibatch,
                          on_enumerate_epoch);
  data_vec.clear();
  label_vec.clear();
  std::cout << "end training." << std::endl;

  // save network model & trained weights
  nn.save("test-model"); //nn.save("some filename",tiny_dnn::content_type::weights_and_model,tiny_dnn::file_format::binary);

}


int main(int argc, char *argv[]) {
  signal(SIGINT, signalHandler);
  double learning_rate                   = 1;
  int epochs                             = 30;
  int minibatch_size                     = 16;
  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();
  //argument initialization
  int arg_ret = init(argc, argv, &learning_rate, &epochs,
                     &minibatch_size, backend_type);
  if(arg_ret) return arg_ret;

  try
  {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    train(std::cin, learning_rate, epochs, minibatch_size, backend_type);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()<< "s.\n";
  }
  catch (tiny_dnn::nn_error &err)
  {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
  return 0;
}
