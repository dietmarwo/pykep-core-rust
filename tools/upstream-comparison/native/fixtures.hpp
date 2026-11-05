// SPDX-License-Identifier: MPL-2.0

#pragma once
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

inline constexpr auto fixture_magic = "PYKEP_BENCH_V1";
inline constexpr std::size_t max_doubles = (128u << 20) / sizeof(double);

inline void write_value(std::ostream &out, double value)
{
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite fixture value");
    auto bits = std::bit_cast<std::uint64_t>(value);
    if constexpr (std::endian::native == std::endian::big) bits = std::byteswap(bits);
    out.write(reinterpret_cast<const char *>(&bits), sizeof(bits));
}

struct Dataset {
    std::string suite;
    std::size_t count{}, width{};
    std::vector<double> data;
    explicit Dataset(const char *path)
    {
        std::ifstream in(path, std::ios::binary);
        std::string magic;
        if (!(in >> magic >> suite >> count >> width) || magic != fixture_magic
            || width == 0 || count == 0 || width > max_doubles / count || in.get() != '\n')
            throw std::runtime_error("invalid fixture header");
        data.resize(count * width);
        for (auto &value : data) {
            std::uint64_t bits;
            if (!in.read(reinterpret_cast<char *>(&bits), sizeof(bits)))
                throw std::runtime_error("truncated fixture");
            if constexpr (std::endian::native == std::endian::big) bits = std::byteswap(bits);
            value = std::bit_cast<double>(bits);
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite fixture value");
        }
        if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing fixture bytes");
    }
    const double *row(std::size_t i) const { return data.data() + i * width; }
};
