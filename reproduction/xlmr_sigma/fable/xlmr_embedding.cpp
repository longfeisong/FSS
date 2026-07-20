#include "GC/emp-sh2pc.h"
#include "GC/lookup.h"
#include "utils/io_utils.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sci;

namespace {

constexpr std::size_t kInputBits = 20;
constexpr std::size_t kElementBits = 16;
constexpr std::size_t kDimensions = 768;
constexpr std::size_t kDimensionsPerChunk = 32;
constexpr std::size_t kNumChunks = kDimensions / kDimensionsPerChunk;
constexpr std::size_t kOutputBits = kElementBits * kDimensionsPerChunk;
constexpr std::size_t kVocabSize = 250002;
constexpr std::size_t kBatchSize = 512;
constexpr std::size_t kSigmaBitwidth = 50;
constexpr uint64_t kSigmaMask = (uint64_t{1} << kSigmaBitwidth) - 1;

int party = 0;
int port = 8000;
int parallel = 1;
int num_threads = 32;
int chunk_start = 0;
int chunk_count = static_cast<int>(kNumChunks);
std::string table_path;
std::string queries_path;
std::string output_path;
NetIO *io_gc = nullptr;

template <typename T>
std::vector<T> read_exact(const std::string &path, std::size_t count) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open " + path);
  }
  std::vector<T> values(count);
  input.read(reinterpret_cast<char *>(values.data()),
             static_cast<std::streamsize>(count * sizeof(T)));
  if (input.gcount() != static_cast<std::streamsize>(count * sizeof(T))) {
    throw std::runtime_error("unexpected size for " + path);
  }
  char extra;
  if (input.read(&extra, 1)) {
    throw std::runtime_error("trailing data in " + path);
  }
  return values;
}

void write_exact(const std::string &path, const std::vector<uint64_t> &values) {
  std::filesystem::path output(path);
  if (output.has_parent_path()) {
    std::filesystem::create_directories(output.parent_path());
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    throw std::runtime_error("cannot create " + path);
  }
  stream.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(uint64_t)));
  if (!stream) {
    throw std::runtime_error("failed to write " + path);
  }
}

rawdatablock pack_chunk(const int16_t *row, std::size_t chunk) {
  rawdatablock packed;
  const std::size_t dimension_offset = chunk * kDimensionsPerChunk;
  for (std::size_t dimension = 0; dimension < kDimensionsPerChunk;
       ++dimension) {
    const uint16_t value =
        static_cast<uint16_t>(row[dimension_offset + dimension]);
    for (std::size_t bit = 0; bit < kElementBits; ++bit) {
      packed[dimension * kElementBits + bit] = (value >> bit) & 1U;
    }
  }
  return packed;
}

std::map<uint64_t, rawdatablock>
make_lut(const std::vector<int16_t> &table, std::size_t chunk) {
  std::map<uint64_t, rawdatablock> lut;
  for (std::size_t token = 0; token < kVocabSize; ++token) {
    lut.emplace(token, pack_chunk(table.data() + token * kDimensions, chunk));
  }
  return lut;
}

std::vector<Integer> make_secret_queries(const std::vector<uint32_t> &queries) {
  std::vector<Integer> secret_queries;
  secret_queries.reserve(kBatchSize);
  for (std::size_t query = 0; query < kBatchSize; ++query) {
    const uint32_t value = party == BOB ? queries[query] : 0;
    secret_queries.emplace_back(kInputBits + 1, value, BOB);
  }
  return secret_queries;
}

void gc_to_arithmetic_shares(const IntegerArray &packed_results,
                             std::size_t chunk,
                             std::vector<uint64_t> &all_shares,
                             osuCrypto::PRNG &share_prng) {
  const std::size_t value_count = kBatchSize * kDimensionsPerChunk;
  const std::size_t total_bits = value_count * kSigmaBitwidth;
  std::vector<uint64_t> alice_shares(value_count, 0);
  std::vector<bool> alice_share_bits_vector(total_bits, false);

  if (party == ALICE) {
    for (std::size_t value_idx = 0; value_idx < value_count; ++value_idx) {
      const uint64_t share = share_prng.get<uint64_t>() & kSigmaMask;
      alice_shares[value_idx] = share;
      for (std::size_t bit = 0; bit < kSigmaBitwidth; ++bit) {
        alice_share_bits_vector[value_idx * kSigmaBitwidth + bit] =
            (share >> bit) & 1U;
      }
    }
  }

  std::unique_ptr<bool[]> alice_share_bits(new bool[total_bits]);
  for (std::size_t bit = 0; bit < total_bits; ++bit) {
    alice_share_bits[bit] = alice_share_bits_vector[bit];
  }
  std::vector<Bit> alice_share_labels(total_bits);
  prot_exec->feed(
      reinterpret_cast<block128 *>(alice_share_labels.data()), ALICE,
      alice_share_bits.get(), static_cast<int>(total_bits));

  std::vector<Bit> bob_share_labels(total_bits);
  for (std::size_t query = 0; query < kBatchSize; ++query) {
    for (std::size_t dimension = 0; dimension < kDimensionsPerChunk;
         ++dimension) {
      const std::size_t value_idx = query * kDimensionsPerChunk + dimension;
      Integer value(kElementBits, 0);
      value.bits.assign(
          packed_results[query].bits.begin() + dimension * kElementBits,
          packed_results[query].bits.begin() + (dimension + 1) * kElementBits);
      value.resize(kSigmaBitwidth, true);

      Integer alice_share(kSigmaBitwidth, 0);
      alice_share.bits.assign(
          alice_share_labels.begin() + value_idx * kSigmaBitwidth,
          alice_share_labels.begin() + (value_idx + 1) * kSigmaBitwidth);
      Integer bob_share = value - alice_share;
      std::copy(bob_share.bits.begin(), bob_share.bits.end(),
                bob_share_labels.begin() + value_idx * kSigmaBitwidth);
    }
  }

  std::unique_ptr<bool[]> bob_share_bits(new bool[total_bits]);
  prot_exec->reveal(bob_share_bits.get(), BOB,
                    reinterpret_cast<block128 *>(bob_share_labels.data()),
                    static_cast<int>(total_bits));

  for (std::size_t query = 0; query < kBatchSize; ++query) {
    for (std::size_t dimension = 0; dimension < kDimensionsPerChunk;
         ++dimension) {
      const std::size_t value_idx = query * kDimensionsPerChunk + dimension;
      uint64_t share = 0;
      if (party == ALICE) {
        share = alice_shares[value_idx];
      } else {
        for (std::size_t bit = 0; bit < kSigmaBitwidth; ++bit) {
          share |= static_cast<uint64_t>(
                       bob_share_bits[value_idx * kSigmaBitwidth + bit])
                   << bit;
        }
      }
      const std::size_t output_dimension =
          chunk * kDimensionsPerChunk + dimension;
      all_shares[query * kDimensions + output_dimension] = share & kSigmaMask;
    }
  }
}

void run() {
  utils::check(LUT_INPUT_SIZE == kInputBits,
               "xlmr_embedding requires LUT_INPUT_SIZE=20");
  utils::check(LUT_OUTPUT_SIZE == kOutputBits,
               "xlmr_embedding requires LUT_OUTPUT_SIZE=512");
  utils::check(chunk_start >= 0 && chunk_count > 0 &&
                   chunk_start + chunk_count <= static_cast<int>(kNumChunks),
               "invalid chunk range");
  utils::check(!output_path.empty(), "output path is required");

  std::vector<int16_t> table;
  if (party == ALICE) {
    utils::check(!table_path.empty(), "ALICE requires table=<path>");
    table = read_exact<int16_t>(table_path, kVocabSize * kDimensions);
  }

  std::vector<uint32_t> queries(kBatchSize, 0);
  if (party == BOB) {
    utils::check(!queries_path.empty(), "BOB requires queries=<path>");
    queries = read_exact<uint32_t>(queries_path, kBatchSize);
    for (uint32_t query : queries) {
      utils::check(query < kVocabSize, "query is outside XLM-R vocabulary");
    }
  }

  std::vector<uint64_t> shares(kBatchSize * kDimensions, 0);
  osuCrypto::PRNG share_prng(osuCrypto::sysRandomSeed());

  for (int chunk = chunk_start; chunk < chunk_start + chunk_count; ++chunk) {
    std::map<uint64_t, rawdatablock> lut;
    if (party == ALICE) {
      lut = make_lut(table, static_cast<std::size_t>(chunk));
    }
    auto secret_queries = make_secret_queries(queries);
    std::cout << "[XLMR] chunk " << chunk + 1 << "/" << kNumChunks
              << ": preparing FABLE" << std::endl;
    auto params = fable_prepare(lut, party, kBatchSize, kVocabSize, parallel,
                                num_threads, BatchPirType::PIRANA,
                                HashType::LowMC, io_gc);
    barrier(party, io_gc);
    io_gc->flush();

    std::cout << "[XLMR] chunk " << chunk + 1 << "/" << kNumChunks
              << ": private lookup" << std::endl;
    auto packed_results = fable_lookup(secret_queries, params, true);

    std::cout << "[XLMR] chunk " << chunk + 1 << "/" << kNumChunks
              << ": GC-to-arithmetic conversion" << std::endl;
    gc_to_arithmetic_shares(packed_results, static_cast<std::size_t>(chunk),
                            shares, share_prng);
    io_gc->flush();
  }

  write_exact(output_path, shares);
  std::cout << "[XLMR] wrote arithmetic share " << output_path << " ("
            << shares.size() * sizeof(uint64_t) << " bytes)" << std::endl;
  std::cout << "[XLMR] FABLE-to-SIGMA share bridge passed" << std::endl;
}

} // namespace

int main(int argc, char **argv) {
  signal(SIGSEGV, handler);
  ArgMapping amap;
  amap.arg("r", party, "Role: ALICE=1 model server, BOB=2 token client");
  amap.arg("p", port, "base port");
  amap.arg("par", parallel, "BatchPIR parallel flag");
  amap.arg("thr", num_threads, "BatchPIR CPU threads");
  amap.arg("chunk_start", chunk_start, "first 32-dimension chunk");
  amap.arg("chunk_count", chunk_count, "number of 32-dimension chunks");
  amap.arg("table", table_path, "row-major int16 XLM-R table");
  amap.arg("queries", queries_path, "512 uint32 token IDs");
  amap.arg("output", output_path, "party arithmetic-share output");
  amap.parse(argc - 1, argv + 1);

  if (party != ALICE && party != BOB) {
    throw std::runtime_error("r must be 1 or 2");
  }
  io_gc = new NetIO(party == ALICE ? nullptr : argv[1],
                    port + GC_PORT_OFFSET, true);
  setup_semi_honest(io_gc, party);
  run();
  delete io_gc;
  return 0;
}
