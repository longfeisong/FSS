#include <sytorch/module.h>
#include <sytorch/utils.h>

#include "experiments/sigma/gpt2.h"
#include "experiments/sigma/bert.h"
#include "backend/sigma.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr u64 kSequenceLength = 128;
constexpr u64 kHiddenSize = 768;
constexpr u64 kHeads = 12;
constexpr int kBitwidth = 50;
constexpr u64 kScale = 12;

void load_sequence(const std::string &path, int sequence_index,
                   Tensor<u64> &input) {
  const auto expected_minimum =
      static_cast<std::uintmax_t>(sequence_index + 1) * input.size() * sizeof(u64);
  if (std::filesystem::file_size(path) < expected_minimum) {
    throw std::runtime_error("input file is too short for selected sequence");
  }
  std::ifstream stream(path, std::ios::binary);
  stream.seekg(static_cast<std::streamoff>(sequence_index) * input.size() *
               sizeof(u64));
  stream.read(reinterpret_cast<char *>(input.data), input.size() * sizeof(u64));
  if (!stream) {
    throw std::runtime_error("failed to read input sequence from " + path);
  }
}

void write_output(const std::string &path, const Tensor<u64> &output) {
  std::filesystem::path destination(path);
  if (destination.has_parent_path()) {
    std::filesystem::create_directories(destination.parent_path());
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char *>(output.data),
               output.size() * sizeof(u64));
  if (!stream) {
    throw std::runtime_error("failed to write " + path);
  }
}

u64 key_buffer_size(int layers) {
  const char *configured = std::getenv("SIGMA_XLMR_KEY_GB");
  u64 gib = configured ? std::stoull(configured)
                       : static_cast<u64>(layers == 1 ? 8 : 6 * layers);
  return gib * OneGB;
}

void usage(const char *program) {
  std::cerr
      << "usage:\n  " << program
      << " dealer <layers> <party> <sequence-index> <dealer-input-mask> "
         "<dealer-weight-mask> <key-prefix>\n  "
      << program
      << " online <layers> <party> <sequence-index> <masked-input> "
         "<masked-weights> <key-prefix> <peer-ip> <threads> <output>\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 8) {
    usage(argv[0]);
    return 2;
  }
  sytorch_init();
  const std::string role(argv[1]);
  const int layers = std::stoi(argv[2]);
  const int party = std::stoi(argv[3]);
  const int sequence_index = std::stoi(argv[4]);
  const std::string input_path(argv[5]);
  const std::string weights_path(argv[6]);
  const std::string key_prefix(argv[7]);
  if (layers < 1 || layers > 12 || (party != 0 && party != 1) ||
      sequence_index < 0 || sequence_index >= 4) {
    throw std::invalid_argument("invalid layers, party, or sequence index");
  }

  Tensor<u64> input({kSequenceLength, kHiddenSize});
  auto net = new GPUBERT<u64>(layers, kHeads, kHiddenSize, "none", "qkvconcat");
  net->init(kScale, input);
  net->loadring(weights_path);
  load_sequence(input_path, sequence_index, input);

  if (role == "dealer") {
    auto backend =
        new SIGMAKeygen<u64>(party, kBitwidth, kScale, key_prefix,
                             key_buffer_size(layers));
    net->setBackend(backend);
    net->optimize();
    input.d_data = (u64 *)moveToGPU((u8 *)input.data,
                                    input.size() * sizeof(u64), nullptr);
    auto &activation = net->forward(input);
    backend->output(activation);
    backend->close();
    std::cout << "SIGMA XLM-R dealer complete: layers=" << layers
              << " party=" << party << " sequence=" << sequence_index
              << " key_bytes=" << backend->keySize << std::endl;
    return 0;
  }

  if (role != "online" || argc != 11) {
    usage(argv[0]);
    return 2;
  }
  const std::string peer_ip(argv[8]);
  const int threads = std::stoi(argv[9]);
  const std::string output_path(argv[10]);
  auto backend = new SIGMA<u64>(party, peer_ip, key_prefix, kBitwidth, kScale,
                                kSequenceLength, kHiddenSize, threads, false);
  net->setBackend(backend);
  net->optimize();
  backend->peer->sync();
  const auto start = std::chrono::high_resolution_clock::now();
  input.d_data = (u64 *)moveToGPU((u8 *)input.data,
                                  input.size() * sizeof(u64), &backend->s);
  auto &activation = net->forward(input);
  backend->output(activation);
  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  const u64 communication =
      backend->peer->bytesSent() + backend->peer->bytesReceived();
  write_output(output_path, activation);
  backend->close();
  std::cout << "SIGMA XLM-R online complete: layers=" << layers
            << " party=" << party << " sequence=" << sequence_index
            << " elapsed_us=" << elapsed << " communication_bytes="
            << communication << " output=" << output_path << std::endl;
  return 0;
}
