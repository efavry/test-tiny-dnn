#include <csignal>

#include "tiny_dnn/tiny_dnn.h"

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

static void signalHandler( int signum )
{
   std::cout << "Interrupt signal (" << signum << ") received.\n" << std::endl;

   exit(signum);
}

static void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn,
                          tiny_dnn::core::backend_t backend_type)
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
