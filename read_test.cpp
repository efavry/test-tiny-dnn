#include <iostream>
#include <string>

#include "tiny_dnn/tiny_dnn.h"

int main(int argc, char *argv[]) {

  std::string line;
  std::string token;
  std::getline(std::cin, line);

  std::istringstream tokenized_line(line);

  tiny_dnn::vec_t data_vec;

  while(std::getline(tokenized_line, token, ' ')) {
    data_vec.push_back(std::stoi(token));
    //std::cout << token << std::endl;
  }
}
