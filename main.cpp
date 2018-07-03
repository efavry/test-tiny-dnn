/*
 * Recopied from tiny dnn here is the original Copyright
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>
#include <chrono>
#include <csignal>

#include "tiny_dnn/tiny_dnn.h"

//#define CNN_TASK_SIZE 32

std::vector<tiny_dnn::vec_t> train_labels={{0, 7, 0, 7, 0, 7, 0, 7},
                                           {0, 3, 0, 7, 0, 7, 0, 7},
                                           {4, 7, 0, 7, 0, 7, 0, 7},
                                           {0, 2, 0, 7, 0, 7, 0, 7},
                                           {3, 5, 0, 7, 0, 7, 0, 7},
                                           {6, 7, 0, 7, 0, 7, 0, 7},
                                           {0, 9, 0, 9, 0, 9, 0, 9},
                                           {0, 4, 0, 9, 0, 9, 0, 9},
                                           {5, 9, 0, 9, 0, 9, 0, 9},
                                           {0, 3, 0, 9, 0, 9, 0, 9},
                                           {4, 6, 0, 9, 0, 9, 0, 9},
                                           {7, 9, 0, 9, 0, 9, 0, 9},
                                           {0, 11, 0, 11, 0, 11, 0, 11},
                                           {0, 5, 0, 11, 0, 11, 0, 11},
                                           {6, 11, 0, 11, 0, 11, 0, 11},
                                           {0, 3, 0, 11, 0, 11, 0, 11},
                                           {4, 7, 0, 11, 0, 11, 0, 11},
                                           {8, 11, 0, 11, 0, 11, 0, 11},
                                           {0, 13, 0, 13, 0, 13, 0, 13},
                                           {0, 6, 0, 13, 0, 13, 0, 13},
                                           {7, 13, 0, 13, 0, 13, 0, 13},
                                           {0, 4, 0, 13, 0, 13, 0, 13},
                                           {5, 9, 0, 13, 0, 13, 0, 13},
                                           {10, 13, 0, 13, 0, 13, 0, 13},
                                           {0, 15, 0, 15, 0, 15, 0, 15},
                                           {0, 7, 0, 15, 0, 15, 0, 15},
                                           {8, 15, 0, 15, 0, 15, 0, 15},
                                           {0, 5, 0, 15, 0, 15, 0, 15},
                                           {6, 10, 0, 15, 0, 15, 0, 15},
                                           {11, 15, 0, 15, 0, 15, 0, 15},
                                           {0, 17, 0, 17, 0, 17, 0, 17},
                                           {0, 8, 0, 17, 0, 17, 0, 17},
                                           {9, 17, 0, 17, 0, 17, 0, 17},
                                           {0, 5, 0, 17, 0, 17, 0, 17},
                                           {6, 11, 0, 17, 0, 17, 0, 17},
                                           {12, 17, 0, 17, 0, 17, 0, 17},
                                           {0, 19, 0, 19, 0, 19, 0, 19},
                                           {0, 9, 0, 19, 0, 19, 0, 19},
                                           {10, 19, 0, 19, 0, 19, 0, 19},
                                           {0, 6, 0, 19, 0, 19, 0, 19},
                                           {7, 13, 0, 19, 0, 19, 0, 19},
                                           {14, 19, 0, 19, 0, 19, 0, 19},
                                           {0, 21, 0, 21, 0, 21, 0, 21},
                                           {0, 10, 0, 21, 0, 21, 0, 21},
                                           {11, 21, 0, 21, 0, 21, 0, 21},
                                           {0, 7, 0, 21, 0, 21, 0, 21},
                                           {8, 14, 0, 21, 0, 21, 0, 21},
                                           {15, 21, 0, 21, 0, 21, 0, 21},
                                           {0, 23, 0, 23, 0, 23, 0, 23},
                                           {0, 11, 0, 23, 0, 23, 0, 23},
                                           {12, 23, 0, 23, 0, 23, 0, 23},
                                           {0, 7, 0, 23, 0, 23, 0, 23},
                                           {8, 15, 0, 23, 0, 23, 0, 23},
                                           {16, 23, 0, 23, 0, 23, 0, 23},
                                           {0, 25, 0, 25, 0, 25, 0, 25},
                                           {0, 12, 0, 25, 0, 25, 0, 25},
                                           {13, 25, 0, 25, 0, 25, 0, 25},
                                           {0, 8, 0, 25, 0, 25, 0, 25},
                                           {9, 17, 0, 25, 0, 25, 0, 25},
                                           {18, 25, 0, 25, 0, 25, 0, 25},
                                           {0, 27, 0, 27, 0, 27, 0, 27},
                                           {0, 13, 0, 27, 0, 27, 0, 27},
                                           {14, 27, 0, 27, 0, 27, 0, 27},
                                           {0, 9, 0, 27, 0, 27, 0, 27},
                                           {10, 18, 0, 27, 0, 27, 0, 27},
                                           {19, 27, 0, 27, 0, 27, 0, 27},
                                           {0, 29, 0, 29, 0, 29, 0, 29},
                                           {0, 14, 0, 29, 0, 29, 0, 29},
                                           {15, 29, 0, 29, 0, 29, 0, 29},
                                           {0, 9, 0, 29, 0, 29, 0, 29},
                                           {10, 19, 0, 29, 0, 29, 0, 29},
                                           {20, 29, 0, 29, 0, 29, 0, 29}};
std::vector<tiny_dnn::vec_t> train_images={{0, 7, 0, 7},
                                           {0, 4, 0, 7},
                                           {3, 7, 0, 7},
                                           {0, 3, 0, 7},
                                           {2, 6, 0, 7},
                                           {5, 7, 0, 7},
                                           {0, 9, 0, 9},
                                           {0, 5, 0, 9},
                                           {4, 9, 0, 9},
                                           {0, 4, 0, 9},
                                           {3, 7, 0, 9},
                                           {6, 9, 0, 9},
                                           {0, 11, 0, 11},
                                           {0, 6, 0, 11},
                                           {5, 11, 0, 11},
                                           {0, 4, 0, 11},
                                           {3, 8, 0, 11},
                                           {7, 11, 0, 11},
                                           {0, 13, 0, 13},
                                           {0, 7, 0, 13},
                                           {6, 13, 0, 13},
                                           {0, 5, 0, 13},
                                           {4, 10, 0, 13},
                                           {9, 13, 0, 13},
                                           {0, 15, 0, 15},
                                           {0, 8, 0, 15},
                                           {7, 15, 0, 15},
                                           {0, 6, 0, 15},
                                           {5, 11, 0, 15},
                                           {10, 15, 0, 15},
                                           {0, 17, 0, 17},
                                           {0, 9, 0, 17},
                                           {8, 17, 0, 17},
                                           {0, 6, 0, 17},
                                           {5, 12, 0, 17},
                                           {11, 17, 0, 17},
                                           {0, 19, 0, 19},
                                           {0, 10, 0, 19},
                                           {9, 19, 0, 19},
                                           {0, 7, 0, 19},
                                           {6, 14, 0, 19},
                                           {13, 19, 0, 19},
                                           {0, 21, 0, 21},
                                           {0, 11, 0, 21},
                                           {10, 21, 0, 21},
                                           {0, 8, 0, 21},
                                           {7, 15, 0, 21},
                                           {14, 21, 0, 21},
                                           {0, 23, 0, 23},
                                           {0, 12, 0, 23},
                                           {11, 23, 0, 23},
                                           {0, 8, 0, 23},
                                           {7, 16, 0, 23},
                                           {15, 23, 0, 23},
                                           {0, 25, 0, 25},
                                           {0, 13, 0, 25},
                                           {12, 25, 0, 25},
                                           {0, 9, 0, 25},
                                           {8, 18, 0, 25},
                                           {17, 25, 0, 25},
                                           {0, 27, 0, 27},
                                           {0, 14, 0, 27},
                                           {13, 27, 0, 27},
                                           {0, 10, 0, 27},
                                           {9, 19, 0, 27},
                                           {18, 27, 0, 27},
                                           {0, 29, 0, 29},
                                           {0, 15, 0, 29},
                                           {14, 29, 0, 29},
                                           {0, 10, 0, 29},
                                           {9, 20, 0, 29},
                                           {19, 29, 0, 29}};



static void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn,tiny_dnn::core::backend_t backend_type)
{

  // construct nets
  //
  // C : convolution
  // S : sub-sampling
  // F : fully connected
  // clang-format off
  using fc = tiny_dnn::layers::fc;//equivalent of dense in keras
  using conv = tiny_dnn::layers::conv;
  using ave_pool = tiny_dnn::layers::ave_pool;
  using tanh = tiny_dnn::activation::tanh;



  using tiny_dnn::core::connection_table;
  using padding = tiny_dnn::padding;

  using leaky_relu = tiny_dnn::activation::leaky_relu;
  using elu = tiny_dnn::activation::elu;

  nn << fc(8, 64, true, backend_type) //8in 64out has bias = true
     << leaky_relu((float_t)1.0) //epsilon = 1.0 in float_t
     << fc(64, 4, true, backend_type) //engin code is 4 for out but with mnist we need something else
     << elu((size_t)4, (float_t)1.0);  // FC 4 out, activation=elu, alpha = 1.0*/

}

static void train(double learning_rate,
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
    // create callback
    auto on_enumerate_epoch = [&]()
    {
        std::cout << std::endl << "Epoch " << epoch << "/" << n_train_epochs << " finished. " << t.elapsed() << "s elapsed." << std::endl;
        ++epoch;
        /*tiny_dnn::result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(train_images.size());
        t.restart();
        if(((float)res.num_success)/res.num_total*100 > 99 )
        {
            std::cout << "99 or up reached !" << epoch << std::endl; //environ 46
        }*/
        std::cout << "-" << std::endl;
        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]()
    {
        disp += n_minibatch;
    };

  // training //train_once exists !
  nn.fit<tiny_dnn::mse>(optimizer,
                          train_images,
                          train_labels,
                          n_minibatch,
                          n_train_epochs,
                          on_enumerate_minibatch,
                          on_enumerate_epoch);

  std::cout << "end training." << std::endl;



  // save network model & trained weights
  nn.save("test-model");

  std::cout << "Testing load" << std::endl;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  nn.load("test-model");
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Loading time " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()<< " microsecondss.\n";

  std::cout << "Testing inference" << std::endl;
  std::vector<tiny_dnn::vec_t> test_images={{0, 7, 0, 7, 0, 7, 0, 7}};
  start = std::chrono::steady_clock::now();
  nn.predict(test_images); //
  end = std::chrono::steady_clock::now();
  std::cout << "Loading time " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()<< " microsecondss.\n";
  // test and show results
  //auto res = nn.test(test_images, test_labels);
  //res.print_detail(std::cout);
  //std::cout << res.num_success << "/" << res.num_total << " and " << ((float)res.num_success)/res.num_total*100  << std::endl;
}

static tiny_dnn::core::backend_t parse_backend_name(const std::string &name)
{
  const std::array<const std::string, 5> names = {{"internal", "nnpack", "libdnn", "avx", "opencl",}};
  for (size_t i = 0; i < names.size(); ++i)
  {
    if (name.compare(names[i]) == 0)
    {
      return static_cast<tiny_dnn::core::backend_t>(i);
    }
  }
  return tiny_dnn::core::default_engine();
}

static void usage(const char *argv0)
{
  std::cout << "Usage: " << argv0 
            << " --learning_rate 1"
            << " --epochs 30"
            << " --minibatch_size 16"
            << " --backend_type internal" << std::endl;
}

void signalHandler( int signum )
{
   std::cout << "Interrupt signal (" << signum << ") received.\n" << std::endl;

   exit(signum);
}


int main(int argc, char **argv)
{
  signal(SIGINT, signalHandler);
  double learning_rate                   = 1;
  int epochs                             = 30;
  int minibatch_size                     = 16;
  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

  if (argc == 2)
  {
    std::string argname(argv[1]);
    if (argname == "--help" || argname == "-h")
    {
      usage(argv[0]);
      return 0;
    }
  }
  for (int count = 1; count + 1 < argc; count += 2) {
    std::string argname(argv[count]);
    if (argname == "--learning_rate")
    {
      learning_rate = atof(argv[count + 1]);
    }
    else if (argname == "--epochs")
    {
      epochs = atoi(argv[count + 1]);
    }
    else if (argname == "--minibatch_size")
    {
      minibatch_size = atoi(argv[count + 1]);
    }
    else if (argname == "--backend_type")
    {
      backend_type = parse_backend_name(argv[count + 1]);
    }
    else
    {
      std::cerr << "Invalid parameter specified - \"" << argname << "\""
                << std::endl;
      usage(argv[0]);
      return -1;
    }
  }
  if (learning_rate <= 0) {
    std::cerr
      << "Invalid learning rate. The learning rate must be greater than 0."
      << std::endl;
    return -1;
  }
  if (epochs <= 0) {
    std::cerr << "Invalid number of epochs. The number of epochs must be "
                 "greater than 0."
              << std::endl;
    return -1;
  }
  if (minibatch_size <= 0 || minibatch_size > 60000)
  {
    std::cerr
      << "Invalid minibatch size. The minibatch size must be greater than 0"
         " and less than dataset size (60000)."
      << std::endl;
    return -1;
  }
  std::cout << "Running with the following parameters:" << std::endl
            << "Learning rate: " << learning_rate << std::endl
            << "Minibatch size: " << minibatch_size << std::endl
            << "Number of epochs: " << epochs << std::endl
            << "Backend type: " << backend_type << std::endl
            << std::endl;
  try
  {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    train(learning_rate, epochs, minibatch_size, backend_type);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()<< "s.\n";
  }
  catch (tiny_dnn::nn_error &err)
  {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
  return 0;
}
