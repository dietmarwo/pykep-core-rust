// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
// Workload distributions follow ESA PyKEP's benchmark descriptions.

#include "fixtures.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <string_view>

#ifdef PYKEP_WITH_SDK
#include <kep3/core_astro/constants.hpp>
#include <kep3/core_astro/ic2par2ic.hpp>
#include <kep3/lambert_problem.hpp>
#include <kep3/udpla/vsop2013.hpp>
#endif

using State = std::array<std::array<double, 3>, 2>;
using Real = std::uniform_real_distribution<double>;
constexpr double pi = std::numbers::pi_v<double>;

struct Writer {
    std::ofstream file;
    Writer(const char *path, const std::string &suite, std::size_t count, std::size_t width)
        : file(path, std::ios::binary)
    {
        if (!file || !count || !width || width > max_doubles / count)
            throw std::runtime_error("invalid output or fixture size");
        file << fixture_magic << ' ' << suite << ' ' << count << ' ' << width << '\n';
    }
    void scalar(double value) { write_value(file, value); }
    template <std::size_t N> void array(const std::array<double,N> &values)
    {
        for (auto v : values) scalar(v);
    }
    void state(const State &value) { array(value[0]); array(value[1]); }
};

State cartesian(const std::array<double,6> &el)
{
#ifdef PYKEP_WITH_SDK
    return kep3::par2ic(el, 1.);
#else
    const auto [a,e,i,node,w,f] = el;
    const double p = a*(1-e*e), cf = std::cos(f), sf = std::sin(f), r = p/(1+e*cf);
    const double co = std::cos(node), so = std::sin(node), cw = std::cos(w), sw = std::sin(w);
    const double ci = std::cos(i), si = std::sin(i), speed = std::sqrt(1./p);
    const std::array<double,3> x{co*cw-so*sw*ci,so*cw+co*sw*ci,sw*si};
    const std::array<double,3> y{-co*sw-so*cw*ci,-so*sw+co*cw*ci,cw*si};
    State out{};
    for (std::size_t j=0; j<3; ++j) {
        out[0][j] = r*(cf*x[j]+sf*y[j]);
        out[1][j] = speed*(-sf*x[j]+(e+cf)*y[j]);
    }
    return out;
#endif
}

void zoh(Writer &out, std::size_t count, std::size_t n)
{
    std::mt19937 rng(424242);
    Real r(-.3,.3), v(-.35,.35), mass(.85,1.15), tof(.3,2.), thrust(0.,.03);
    for (std::size_t row=0; row<count; ++row) {
        for (int i=0; i<2; ++i) out.array(std::array{1.+r(rng),r(rng),r(rng),v(rng),1.+v(rng),v(rng),mass(rng)});
        const double dt = tof(rng)/n;
        for (std::size_t j=0; j<=n; ++j) out.scalar(j*dt);
        for (std::size_t j=0; j<n; ++j) {
            std::normal_distribution<double> normal(0.,1.);
            std::array<double,3> dir{normal(rng),normal(rng),normal(rng)};
            const auto norm = std::sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
            if (norm <= std::numeric_limits<double>::epsilon()) dir = {1., 0., 0.};
            else for (auto &d : dir) d /= norm;
            out.scalar(thrust(rng));
            out.array(dir);
        }
    }
}

void sf_simple(Writer &out, std::size_t count)
{
    for (std::size_t i=0; i<count; ++i) {
        out.array(std::array{1.,.1,-.1,.2,1.,-.2,1.,1.2,-.1,.1,-.2,1.023,-.44,13./15.,1.,1.,9.80665,1.,.5});
        for (int j=0; j<15; ++j) out.scalar(.1);
    }
}

#ifdef PYKEP_WITH_SDK
void sf(Writer &out, std::size_t count, std::size_t n, bool convergence)
{
    std::mt19937 rng(122012203u), population(424242u);
    Real perturb(0.,.1), mass(1.,1.2), tof(pi/12.,2*pi), epoch(2170.,2200.), throttle(-1.,1.), final_mass(.5,1.);
    const kep3::udpla::vsop2013 earth("earth_moon", 1e-2), jupiter("jupiter", 1e-2);
    State start{}, end{};
    double ms = 1.0;
    for (std::size_t i=0; i<count; ++i) {
        if (i % (convergence ? 1 : 10) == 0) {
            start = earth.eph(epoch(rng));
            end = jupiter.eph(epoch(rng));
            const double guess = tof(rng);
            for (auto *p : {&start,&end}) for (auto &v : (*p)[0]) v /= kep3::AU;
            const kep3::lambert_problem lambert(start[0],end[0],guess,1.);
            start[1] = lambert.get_v0()[0];
            end[1] = lambert.get_v1()[0];
            for (auto &v : end[1]) v += perturb(rng);
            ms = mass(rng);
        }
        out.state(start); out.scalar(ms); out.state(end); out.scalar(final_mass(population));
        out.array(std::array{tof(population),1.,kep3::G0,1.,.5});
        for (std::size_t j=0; j<3*n; ++j) out.scalar(throttle(population));
    }
}
#endif

int main(int argc, char **argv)
{
    try {
        if (argc != 4) throw std::runtime_error("usage: export-fixtures SUITE COUNT OUTPUT");
        const std::string suite = argv[1];
        const auto count = std::stoull(argv[2]);
        std::mt19937 rng(122012203u);
        if (suite.starts_with("anomaly")) {
            const auto band = std::stoul(suite.substr(7,1));
            const std::array<double,4> edges{0.,.5,.9,.99};
            if (band >= edges.size()-1) throw std::runtime_error("unknown anomaly band");
            Real mean(-1e8,1e8), e(edges[band],edges[band+1]);
            Writer out(argv[3],suite,count,2);
            for (std::size_t i=0; i<count; ++i) out.array(std::array{mean(rng),e(rng)});
        } else if (suite.starts_with("lagrangian") || suite.starts_with("roundtrip") || suite.starts_with("stm")) {
            const auto band = std::stoul(suite.substr(suite.size()-1));
            const bool accuracy = suite.starts_with("roundtrip");
            const bool stm = suite.starts_with("stm");
            const std::array<std::array<double,2>,4> bands{{{0.,.5},{.5,.9},{.9,.99},{1.1,10.}}};
            auto [lo,hi] = bands.at(band);
            if (stm && band == 2) { lo=1.1; hi=2.; }
            if (accuracy) rng.seed(53534535u);
            Real e(lo,hi), a(.5,accuracy?10.:20.), theta(0.,2*pi), incl(0.,pi), tof(10.,100.);
            Writer out(argv[3],suite,count,7);
            for (std::size_t i=0; i<count; ++i) {
                double f=pi, eccentricity, sma;
                if (accuracy) {
                    do {
                        eccentricity=e(rng);
                        sma=a(rng);
                        if (eccentricity>1.) sma=-sma;
                        f=theta(rng);
                    } while (sma<0. && std::cos(f)<-1./eccentricity);
                } else {
                    eccentricity=e(rng); sma=a(rng);
                    if (eccentricity>1.) sma=-sma;
                    while (sma<0. && std::cos(f)<-1./eccentricity) f=theta(rng);
                }
                out.state(cartesian({sma,eccentricity,incl(rng),theta(rng),theta(rng),f}));
                out.scalar(tof(rng));
            }
        } else if (suite == "lambert0" || suite == "lambert20") {
            Real r(-2.,2.), tof(2.,40.), mu(.9,1.1);
            std::uniform_int_distribution<unsigned> clockwise(0,1);
            Writer out(argv[3],suite,count,9);
            for (std::size_t i=0; i<count; ++i)
                out.array(std::array{r(rng),r(rng),r(rng),r(rng),r(rng),r(rng),tof(rng),double(clockwise(rng)),mu(rng)});
        } else if (suite.starts_with("zoh")) {
            const auto n=std::stoul(suite.substr(3));
            if (n < 1 || n > 512) throw std::runtime_error("invalid segment count");
            Writer out(argv[3],suite,count,15+5*n);
            zoh(out,count,n);
        } else if (suite == "sf-simple") {
            Writer out(argv[3],suite,count,34);
            sf_simple(out,count);
#ifdef PYKEP_WITH_SDK
        } else if (suite.starts_with("sf")) {
            const bool convergence = suite.starts_with("sf-conv");
            const auto n=std::stoul(suite.substr(convergence ? 7 : 2));
            if (n < 1 || n > 512) throw std::runtime_error("invalid segment count");
            Writer out(argv[3],suite,count,19+3*n);
            sf(out,count,n,convergence);
#endif
        } else {
            throw std::runtime_error("unknown suite or SDK required for SF sampling");
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
