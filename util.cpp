#include <csignal>

#include "tiny_dnn/tiny_dnn.h"

static void append_to_vec_from_ssv(tiny_dnn::vec_t &vec,
                                   const std::string &ssv) {
  std::istringstream streamized_line(ssv);
  std::string token;
  while(std::getline(streamized_line, token, ' ')) {
    vec.push_back(std::stoi(token));
  }
}

static inline tiny_dnn::vec_t vec_from_ssv(const std::string &ssv) {
  tiny_dnn::vec_t vec;
  append_to_vec_from_ssv(vec, ssv);
  return vec;
}

static void parse_and_append(std::string &line,
                             std::vector<tiny_dnn::vec_t> &data_vec,
                             std::vector<tiny_dnn::vec_t> &label_vec) {
  std::istringstream streamized_line(line);
  std::string data, label;
  std::getline(streamized_line, data, ':');
  std::getline(streamized_line, label, ':');

  data_vec.push_back(vec_from_ssv(data));
  label_vec.push_back(vec_from_ssv(label));
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

static int init(int argc, char* argv[],
                 double *learning_rate, int *epochs,
                 int *minibatch_size,
                 tiny_dnn::core::backend_t &backend_type) {
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
      *learning_rate = atof(argv[count + 1]);
    }
    else if (argname == "--epochs")
    {
      *epochs = atoi(argv[count + 1]);
    }
    else if (argname == "--minibatch_size")
    {
      *minibatch_size = atoi(argv[count + 1]);
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
  if (*learning_rate <= 0) {
    std::cerr
      << "Invalid learning rate. The learning rate must be greater than 0."
      << std::endl;
    return -1;
  }
  if (*epochs <= 0) {
    std::cerr << "Invalid number of epochs. The number of epochs must be "
                 "greater than 0."
              << std::endl;
    return -1;
  }
  if (*minibatch_size <= 0 || *minibatch_size > 60000)
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
  return 0;

}
