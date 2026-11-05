// SPDX-License-Identifier: MPL-2.0
// Developer-only benchmark against an installed PyKEP C++ SDK.

#include <kep3/ta/bcp.hpp>
#include <kep3/ta/cr3bp.hpp>
#include <kep3/ta/kep.hpp>
#include <kep3/ta/pontryagin_cartesian.hpp>
#include <kep3/ta/pontryagin_equinoctial.hpp>
#include <kep3/ta/zoh_cr3bp.hpp>
#include <kep3/ta/zoh_eq.hpp>
#include <kep3/ta/zoh_kep.hpp>
#include <kep3/ta/zoh_ss.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using Integrator = heyoka::taylor_adaptive<double>;
using Clock = std::chrono::steady_clock;

const Integrator &factory(const std::string &name, double tolerance)
{
    using namespace kep3::ta;
    if (name == "kepler") return get_ta_kep(tolerance);
    if (name == "cr3bp") return get_ta_cr3bp(tolerance);
    if (name == "bcp") return get_ta_bcp(tolerance);
    if (name == "zoh_kepler") return get_ta_zoh_kep(tolerance);
    if (name == "zoh_cr3bp") return get_ta_zoh_cr3bp(tolerance);
    if (name == "zoh_equinoctial") return get_ta_zoh_eq(tolerance);
    if (name == "solar_sail") return get_ta_zoh_ss(tolerance);
    const auto objective = name.ends_with("_mass") ? kep3::MASS : kep3::TIME;
    if (name == "cartesian_mass" || name == "cartesian_time") return get_ta_pc(tolerance, objective);
    if (name == "equinoctial_mass" || name == "equinoctial_time") return get_ta_peq(tolerance, objective);
    throw std::runtime_error("unknown model: " + name);
}

std::vector<double> read_vector(std::istream &input, std::size_t size)
{
    std::vector<double> values(size);
    for (double &value : values) {
        if (!(input >> value) || !std::isfinite(value)) throw std::runtime_error("invalid fixture value");
    }
    return values;
}

template <typename Function>
std::array<double, 7> measure(Function &&function, std::size_t repeats)
{
    std::array<double, 7> times{};
    for (double &time : times) {
        const auto start = Clock::now();
        for (std::size_t i = 0; i < repeats; ++i) function();
        time = std::chrono::duration<double, std::micro>(Clock::now() - start).count() / repeats;
    }
    return times;
}

double median(std::array<double, 7> times)
{
    std::sort(times.begin(), times.end());
    return times[3];
}

int main(int argc, char **argv)
{
    try {
        if (argc < 2) throw std::runtime_error(
            "usage: pykep-native-comparison FIXTURES.txt [--tolerance=T] [--repeat-scale=N]");
        std::optional<double> selected_tolerance;
        std::size_t repeat_scale = 1;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option.starts_with("--tolerance=")) selected_tolerance = std::stod(option.substr(12));
            else if (option.starts_with("--repeat-scale=")) repeat_scale = std::stoull(option.substr(15));
            else throw std::runtime_error("unknown option: " + option);
        }
        if (!repeat_scale || repeat_scale > 32
            || (selected_tolerance && *selected_tolerance != 1e-9 && *selected_tolerance != 1e-12
                && *selected_tolerance != 1e-14))
            throw std::runtime_error("invalid tolerance or repeat scale");
        std::ifstream input(argv[1]);
        if (!input) throw std::runtime_error("cannot open fixture file");
        std::cout << "model,method,tolerance,min_us,median_us,max_us,scaled_endpoint_error,steps,order,"
                     "setup_us,first_call_us,reset_us,high_accuracy,compact_mode,samples_us\n"
                  << std::setprecision(17);
        std::string name;
        while (input >> name) {
            double start{}, end{};
            std::size_t states{}, parameters{};
            if (!(input >> start >> end >> states >> parameters) || states == 0 || states > 14 || parameters > 32) {
                throw std::runtime_error("invalid fixture header for " + name);
            }
            const auto initial = read_vector(input, states);
            const auto pars = read_vector(input, parameters);
            const auto reference = read_vector(input, states);
            for (const double tolerance : {1e-9, 1e-12, 1e-14}) {
                if (selected_tolerance && *selected_tolerance != tolerance) continue;
                const auto setup_start = Clock::now();
                auto integrator = factory(name, tolerance);
                const double setup = std::chrono::duration<double, std::micro>(Clock::now() - setup_start).count();
                if (integrator.get_dim() != states || integrator.get_pars().size() != parameters) {
                    throw std::runtime_error("fixture dimensions do not match " + name);
                }
                std::copy(pars.begin(), pars.end(), integrator.get_pars_data());
                const auto reset = [&] {
                    integrator.set_time(start);
                    std::copy(initial.begin(), initial.end(), integrator.get_state_data());
                };
                const auto propagate = [&] {
                    reset();
                    const auto result = integrator.propagate_until(end, heyoka::kw::max_steps = 100000);
                    if (std::get<0>(result) != heyoka::taylor_outcome::time_limit) {
                        throw std::runtime_error("propagation failed for " + name);
                    }
                    return std::get<3>(result);
                };
                const auto first_start = Clock::now();
                const auto steps = propagate();
                const double first = std::chrono::duration<double, std::micro>(Clock::now() - first_start).count();
                double error = 0;
                for (std::size_t i = 0; i < states; ++i) {
                    const auto value = integrator.get_state_data()[i];
                    if (!std::isfinite(value)) throw std::runtime_error("nonfinite state for " + name);
                    error = std::max(error, std::abs(value - reference[i]) / std::max(1., std::abs(reference[i])));
                }
                const auto times = measure(propagate, 256 * repeat_scale);
                const auto reset_times = measure(reset, 256 * repeat_scale);
                const auto [min, max] = std::minmax_element(times.begin(), times.end());
                std::cout << name << ",upstream," << tolerance << ',' << *min << ',' << median(times) << ',' << *max
                          << ',' << error << ',' << steps << ',' << integrator.get_order() << ',' << setup << ',' << first
                          << ',' << median(reset_times) << ',' << integrator.get_high_accuracy() << ','
                          << integrator.get_compact_mode() << ',';
                for (std::size_t i = 0; i < times.size(); ++i) std::cout << (i == 0 ? "" : "|") << times[i];
                std::cout << '\n';
            }
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
