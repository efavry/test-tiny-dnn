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
#include "hyper_rect_ops.hpp"

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

  //tiny_dnn::progress_display disp(48);
  tiny_dnn::timer t;

  optimizer.alpha *=
    std::min(tiny_dnn::float_t(4),
             static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

  int epoch = 1;
  int num_batches = 1;
  // create callback
  auto on_enumerate_minibatch = [&]()
  {
      //disp += n_minibatch;
  };

  // training //train_once exists !
  std::string line;
  std::vector<tiny_dnn::vec_t> data_vec;
  std::vector<tiny_dnn::vec_t> label_vec;

  std::vector<tiny_dnn::vec_t> training_data;
  std::vector<tiny_dnn::vec_t> training_labels;
  
  std::vector<tiny_dnn::vec_t> val_data;
  std::vector<tiny_dnn::vec_t> val_labels;

  std::ofstream log_stream;
  log_stream.open("training_log");

  //if the learning rate is <1.0 then this metric is meaningless. So
  //init it to something meaningless
  double current_accuracy = -1.0;

  auto on_epoch = [&]() {
    if(learning_rate < 1) {
      //std::cerr << "Epoch ended" << std::endl;
      std::cout << std::endl <<
                   "Batch " << num_batches << "-" <<
                   "Epoch " << epoch << "/" << n_train_epochs <<
                   " finished." << std::endl;
      epoch += 1;
      size_t total = val_data.size();
      size_t wrong = 0;
      float perf_eff = 0.0, mem_eff = 0.0;
      for(size_t i = 0 ; i < total ; i++) {
        tiny_dnn::vec_t res = nn.predict(val_data[i]);
        tiny_dnn::vec_t whole = get_whole_from_data(val_data[i]);
        auto norm_res = normalize_prediction(res, whole);
        perf_eff += perf_efficiency(val_labels[i], norm_res);
        mem_eff += mem_efficiency(val_labels[i], norm_res);
        std::cout << "\t\tCalculated mem efficiency " << mem_eff << std::endl;
        for(size_t j = 0 ; j < res.size() ; j++) {
          if(norm_res[j] != val_labels[i][j]) {
            wrong += 1;
            break;
          }
        }
      }
      current_accuracy = 1.0*(total-wrong)/total;
      std::cout << "Validation accuracy : " << wrong << "/" << total <<
                   "=" << current_accuracy << std::endl;
      std::cout << "\tPerf Efficiency:" << perf_eff/total << std::endl;
      std::cout << "\tMem Efficiency:" << mem_eff/total << std::endl;
    }
  };

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

      size_t training_size = data_vec.size() * learning_rate;
      size_t validation_size = data_vec.size() - training_size;

      // sigh.. I miss C arrays
      // at the very least don't reallocate these
      // ideally try and see if c++ stl can guarantee contigious storage
      // and copy some how. I don't ,ind using std::vector.data() and
      // play with C pointers, at all
      training_data =
          std::vector<tiny_dnn::vec_t>(data_vec.begin(), 
                                       data_vec.begin()+training_size);
      training_labels =
          std::vector<tiny_dnn::vec_t>(label_vec.begin(),
                                       label_vec.begin()+training_size);

      if(learning_rate < 1) {
        val_data = std::vector<tiny_dnn::vec_t>(data_vec.begin()+
                                                  training_size+1,
                                                data_vec.end());
        val_labels =
            std::vector<tiny_dnn::vec_t>(label_vec.begin()+
                                            training_size+1,
                                         label_vec.end());
      }


      try {
        nn.fit<tiny_dnn::custom>(optimizer,
                              training_data,
                              training_labels,
                              n_minibatch,
                              n_train_epochs,
                              on_enumerate_minibatch,
                              on_epoch);
      }
      catch (tiny_dnn::nn_error &err) {
        std::cerr << "Exception: " << err.what() << std::endl;
      }
      

      data_vec.clear();
      label_vec.clear();
      epoch = 1;
      num_batches += 1;
    }
    else {
      parse_and_append(line, data_vec, label_vec);
      //std::cout << "Parsed and appended " << line << std::endl;
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
  // NOTE: currently, because of the way the python framework uses
  // this executable, this shouldn't happen. but I am still keeping
  // the logic just in case
  if(data_vec.size() > 0) {
    std::cout << "Outside Fit called with " << data_vec.size() <<
      " data point scanned\n";

    nn.fit<tiny_dnn::custom>(optimizer,
        data_vec,
        label_vec,
        n_minibatch,
        n_train_epochs,
        on_enumerate_minibatch,
        on_epoch);
    data_vec.clear();
    label_vec.clear();
  }
  std::cout << "end training." << std::endl;
  log_stream.close();
  std::getline(data_stream, line);
  std::cout << "Saving model : " << line << std::endl;
  std::cout << "Final accuracy : " << current_accuracy << std::endl;

  // save network model & trained weights
  nn.save(line);
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
    auto start = get_time();

    train(std::cin, learning_rate, epochs, minibatch_size, backend_type);

    auto end = get_time();
    std::cout << "Time " << time_diff(end, start) << "s.\n";
  }
  catch (tiny_dnn::nn_error &err)
  {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
  return 0;
}
