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

//std::vector<tiny_dnn::vec_t> train_labels={{0, 7, 0, 7, 0, 7, 0, 7},
                                           //{0, 3, 0, 7, 0, 7, 0, 7},
                                           //{4, 7, 0, 7, 0, 7, 0, 7},
                                           //{0, 2, 0, 7, 0, 7, 0, 7},
                                           //{3, 5, 0, 7, 0, 7, 0, 7},
                                           //{6, 7, 0, 7, 0, 7, 0, 7},
                                           //{0, 9, 0, 9, 0, 9, 0, 9},
                                           //{0, 4, 0, 9, 0, 9, 0, 9},
                                           //{5, 9, 0, 9, 0, 9, 0, 9},
                                           //{0, 3, 0, 9, 0, 9, 0, 9},
                                           //{4, 6, 0, 9, 0, 9, 0, 9},
                                           //{7, 9, 0, 9, 0, 9, 0, 9},
                                           //{0, 11, 0, 11, 0, 11, 0, 11},
                                           //{0, 5, 0, 11, 0, 11, 0, 11},
                                           //{6, 11, 0, 11, 0, 11, 0, 11},
                                           //{0, 3, 0, 11, 0, 11, 0, 11},
                                           //{4, 7, 0, 11, 0, 11, 0, 11},
                                           //{8, 11, 0, 11, 0, 11, 0, 11},
                                           //{0, 13, 0, 13, 0, 13, 0, 13},
                                           //{0, 6, 0, 13, 0, 13, 0, 13},
                                           //{7, 13, 0, 13, 0, 13, 0, 13},
                                           //{0, 4, 0, 13, 0, 13, 0, 13},
                                           //{5, 9, 0, 13, 0, 13, 0, 13},
                                           //{10, 13, 0, 13, 0, 13, 0, 13},
                                           //{0, 15, 0, 15, 0, 15, 0, 15},
                                           //{0, 7, 0, 15, 0, 15, 0, 15},
                                           //{8, 15, 0, 15, 0, 15, 0, 15},
                                           //{0, 5, 0, 15, 0, 15, 0, 15},
                                           //{6, 10, 0, 15, 0, 15, 0, 15},
                                           //{11, 15, 0, 15, 0, 15, 0, 15},
                                           //{0, 17, 0, 17, 0, 17, 0, 17},
                                           //{0, 8, 0, 17, 0, 17, 0, 17},
                                           //{9, 17, 0, 17, 0, 17, 0, 17},
                                           //{0, 5, 0, 17, 0, 17, 0, 17},
                                           //{6, 11, 0, 17, 0, 17, 0, 17},
                                           //{12, 17, 0, 17, 0, 17, 0, 17},
                                           //{0, 19, 0, 19, 0, 19, 0, 19},
                                           //{0, 9, 0, 19, 0, 19, 0, 19},
                                           //{10, 19, 0, 19, 0, 19, 0, 19},
                                           //{0, 6, 0, 19, 0, 19, 0, 19},
                                           //{7, 13, 0, 19, 0, 19, 0, 19},
                                           //{14, 19, 0, 19, 0, 19, 0, 19},
                                           //{0, 21, 0, 21, 0, 21, 0, 21},
                                           //{0, 10, 0, 21, 0, 21, 0, 21},
                                           //{11, 21, 0, 21, 0, 21, 0, 21},
                                           //{0, 7, 0, 21, 0, 21, 0, 21},
                                           //{8, 14, 0, 21, 0, 21, 0, 21},
                                           //{15, 21, 0, 21, 0, 21, 0, 21},
                                           //{0, 23, 0, 23, 0, 23, 0, 23},
                                           //{0, 11, 0, 23, 0, 23, 0, 23},
                                           //{12, 23, 0, 23, 0, 23, 0, 23},
                                           //{0, 7, 0, 23, 0, 23, 0, 23},
                                           //{8, 15, 0, 23, 0, 23, 0, 23},
                                           //{16, 23, 0, 23, 0, 23, 0, 23},
                                           //{0, 25, 0, 25, 0, 25, 0, 25},
                                           //{0, 12, 0, 25, 0, 25, 0, 25},
                                           //{13, 25, 0, 25, 0, 25, 0, 25},
                                           //{0, 8, 0, 25, 0, 25, 0, 25},
                                           //{9, 17, 0, 25, 0, 25, 0, 25},
                                           //{18, 25, 0, 25, 0, 25, 0, 25},
                                           //{0, 27, 0, 27, 0, 27, 0, 27},
                                           //{0, 13, 0, 27, 0, 27, 0, 27},
                                           //{14, 27, 0, 27, 0, 27, 0, 27},
                                           //{0, 9, 0, 27, 0, 27, 0, 27},
                                           //{10, 18, 0, 27, 0, 27, 0, 27},
                                           //{19, 27, 0, 27, 0, 27, 0, 27},
                                           //{0, 29, 0, 29, 0, 29, 0, 29},
                                           //{0, 14, 0, 29, 0, 29, 0, 29},
                                           //{15, 29, 0, 29, 0, 29, 0, 29},
                                           //{0, 9, 0, 29, 0, 29, 0, 29},
                                           //{10, 19, 0, 29, 0, 29, 0, 29},
                                           //{20, 29, 0, 29, 0, 29, 0, 29}};
//std::vector<tiny_dnn::vec_t> train_images={{0, 7, 0, 7},
                                           //{0, 4, 0, 7},
                                           //{3, 7, 0, 7},
                                           //{0, 3, 0, 7},
                                           //{2, 6, 0, 7},
                                           //{5, 7, 0, 7},
                                           //{0, 9, 0, 9},
                                           //{0, 5, 0, 9},
                                           //{4, 9, 0, 9},
                                           //{0, 4, 0, 9},
                                           //{3, 7, 0, 9},
                                           //{6, 9, 0, 9},
                                           //{0, 11, 0, 11},
                                           //{0, 6, 0, 11},
                                           //{5, 11, 0, 11},
                                           //{0, 4, 0, 11},
                                           //{3, 8, 0, 11},
                                           //{7, 11, 0, 11},
                                           //{0, 13, 0, 13},
                                           //{0, 7, 0, 13},
                                           //{6, 13, 0, 13},
                                           //{0, 5, 0, 13},
                                           //{4, 10, 0, 13},
                                           //{9, 13, 0, 13},
                                           //{0, 15, 0, 15},
                                           //{0, 8, 0, 15},
                                           //{7, 15, 0, 15},
                                           //{0, 6, 0, 15},
                                           //{5, 11, 0, 15},
                                           //{10, 15, 0, 15},
                                           //{0, 17, 0, 17},
                                           //{0, 9, 0, 17},
                                           //{8, 17, 0, 17},
                                           //{0, 6, 0, 17},
                                           //{5, 12, 0, 17},
                                           //{11, 17, 0, 17},
                                           //{0, 19, 0, 19},
                                           //{0, 10, 0, 19},
                                           //{9, 19, 0, 19},
                                           //{0, 7, 0, 19},
                                           //{6, 14, 0, 19},
                                           //{13, 19, 0, 19},
                                           //{0, 21, 0, 21},
                                           //{0, 11, 0, 21},
                                           //{10, 21, 0, 21},
                                           //{0, 8, 0, 21},
                                           //{7, 15, 0, 21},
                                           //{14, 21, 0, 21},
                                           //{0, 23, 0, 23},
                                           //{0, 12, 0, 23},
                                           //{11, 23, 0, 23},
                                           //{0, 8, 0, 23},
                                           //{7, 16, 0, 23},
                                           //{15, 23, 0, 23},
                                           //{0, 25, 0, 25},
                                           //{0, 13, 0, 25},
                                           //{12, 25, 0, 25},
                                           //{0, 9, 0, 25},
                                           //{8, 18, 0, 25},
                                           //{17, 25, 0, 25},
                                           //{0, 27, 0, 27},
                                           //{0, 14, 0, 27},
                                           //{13, 27, 0, 27},
                                           //{0, 10, 0, 27},
                                           //{9, 19, 0, 27},
                                           //{18, 27, 0, 27},
                                           //{0, 29, 0, 29},
                                           //{0, 15, 0, 29},
                                           //{14, 29, 0, 29},
                                           //{0, 10, 0, 29},
                                           //{9, 20, 0, 29},
                                           //{19, 29, 0, 29}};




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

  std::cout << "load models..." << std::endl;


  std::cout << "start training" << std::endl;

  tiny_dnn::progress_display disp(train_images.size());
  tiny_dnn::timer t;

  optimizer.alpha *=
    std::min(tiny_dnn::float_t(4),
             static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

  int epoch = 1;
  //
  // create callback
  auto on_enumerate_epoch = [&]()
  {
      std::cout << std::endl <<
        "Epoch " << epoch << "/" << n_train_epochs << " finished. " <<
        t.elapsed() << "s elapsed." << std::endl;
      ++epoch;

      std::cout << "-" << std::endl;
      disp.restart(train_images.size());
      t.restart();
  };

  auto on_enumerate_minibatch = [&]()
  {
      disp += n_minibatch;
  };

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
      std::cout << "Here\n";
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
  nn.save("test-model");

  std::cout << "Testing load" << std::endl;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  nn.load("test-model");
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Loading time " << 
    std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() <<
    " microsecondss.\n";

  std::cout << "Testing inference" << std::endl;
  std::vector<tiny_dnn::vec_t> test_images={{0, 7, 0, 7, 0, 7, 0, 7}};
  start = std::chrono::steady_clock::now();
  nn.predict(test_images); //
  end = std::chrono::steady_clock::now();
  std::cout << "Loading time " <<
    std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()<<
    " microsecondss.\n";
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
