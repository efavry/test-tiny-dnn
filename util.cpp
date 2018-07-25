#include <csignal>
#include <chrono>

#include "tiny_dnn/tiny_dnn.h"

tiny_dnn::vec_t get_whole_from_data(const tiny_dnn::vec_t &data) {
  tiny_dnn::vec_t res;
  for(size_t i = data.size()/2 ; i < data.size() ; i++) {
    res.push_back(data[i]);
  }
  return res;
}


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

static inline std::chrono::steady_clock::time_point get_time() {
  return std::chrono::steady_clock::now();
}

static inline auto time_diff(std::chrono::steady_clock::time_point &end,
                             std::chrono::steady_clock::time_point &start) {
  return std::chrono::duration_cast<std::chrono::seconds>(end-start).count();
}

static void append_to_vec_from_ssv(tiny_dnn::vec_t &vec,
                                   const std::string &ssv) {
  std::istringstream streamized_line(ssv);
  std::string token;
  //std::cout << "Line : " << ssv << std::endl;
  try {
    while(std::getline(streamized_line, token, ' ')) {
      //std::cout << "Token : " << token << std::endl;
      vec.push_back(std::stoi(token));
    }
  }
  catch(std::invalid_argument e) {
    std::cout << "Invalid argument for " << ssv << std::endl;
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
                          tiny_dnn::core::backend_t backend_type,
                          int in_arr_dim)
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

  int in_size = in_arr_dim*4;
  int mid_size = in_size*in_size;
  int out_size = in_arr_dim*2;
  nn << fc(in_size, mid_size, true, backend_type) //8in 64out has bias = true
     << leaky_relu((float_t)1.0) //epsilon = 1.0 in float_t
     << fc(mid_size, out_size, true, backend_type) //engin code is 4 for out but with mnist we need something else
     << elu((size_t)4, (float_t)1.0);  // FC 4 out, activation=elu, alpha = 1.0*/
}

static int init(int argc, char* argv[],
                 double *learning_rate, int *epochs,
                 int *minibatch_size, int *in_arr_dim,
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
    else if (argname == "--in-arr-dim")
    {
      *in_arr_dim = atoi(argv[count + 1]);
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
  std::cerr << "Running with the following parameters:" << std::endl
            << "Learning rate: " << *learning_rate << std::endl
            << "Minibatch size: " << *minibatch_size << std::endl
            << "Number of epochs: " << *epochs << std::endl
            << "Backend type: " << backend_type << std::endl
            << std::endl;
  return 0;

}
