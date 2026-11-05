// SPDX-License-Identifier: MPL-2.0

#include "fixtures.hpp"
#include <kep3/core_astro/convert_anomalies.hpp>
#include <kep3/core_astro/propagate_lagrangian.hpp>
#include <kep3/lambert_problem.hpp>
#include <kep3/leg/sims_flanagan.hpp>
#include <kep3/leg/zoh.hpp>
#include <kep3/ta/zoh_kep.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <span>
#include <tuple>

using Clock = std::chrono::steady_clock;
using State = std::array<std::array<double,3>,2>;

struct Options { std::size_t passes=5, repetitions=1, values=0; double derivative_tol=1e-8; };

template<class T>
void flatten(std::vector<double> &out, const T &value)
{
    if constexpr (std::ranges::range<T>) {
        for (const auto &entry : value) flatten(out,entry);
    } else {
        out.push_back(value);
    }
}

template<class... T>
std::vector<double> flat(const T &...value)
{
    std::vector<double> out;
    (flatten(out,value),...);
    return out;
}

void array(const std::vector<double> &values)
{
    std::cout << '[';
    for (std::size_t i=0; i<values.size(); ++i) {
        if (i) std::cout << ',';
        if (std::isfinite(values[i])) std::cout << values[i]; else std::cout << "null";
    }
    std::cout << ']';
}

template<class T>
inline void consume(const T &value)
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(&value) : "memory");
#else
    const T *volatile sink = &value;
    (void)sink;
#endif
}

template<class Run, class View>
void measure(const Dataset &data, const char *name, const Options &opt, Run &&run, View &&view)
{
    const auto prefix = "{\"suite\":\"" + data.suite + "\",\"operation\":\"" + name + "\"";
    std::size_t failures=0, scalars=0;
    double max_abs=0., sum_abs=0.;
    for (std::size_t i=0; i<data.count; ++i) {
        try {
            const auto values = view(run(i));
            if (!std::ranges::all_of(values,[](double v){ return std::isfinite(v); }))
                throw std::runtime_error("nonfinite result");
            scalars += values.size();
            for (double v : values) {
                max_abs=std::max(max_abs,std::abs(v)); sum_abs+=std::abs(v);
            }
            if (i<opt.values) {
                std::cout << prefix << ",\"case\":" << i << ",\"values\":";
                array(values);
                std::cout << "}\n";
            }
        } catch (const std::exception &e) {
            if (++failures<=8)
                std::cout << prefix << ",\"case\":" << i << ",\"error\":" << std::quoted(e.what()) << "}\n";
        }
    }
    std::vector<double> samples;
    for (std::size_t pass=0; pass<opt.passes; ++pass) {
        const auto start=Clock::now();
        for (std::size_t r=0; r<opt.repetitions; ++r)
            for (std::size_t i=0; i<data.count; ++i) {
                try { consume(run(i)); } catch (const std::exception &) {}
            }
        samples.push_back(std::chrono::duration<double,std::nano>(Clock::now()-start).count()
                          / (data.count * opt.repetitions));
    }
    std::cout << prefix << ",\"cases\":" << data.count << ",\"failed\":" << failures
              << ",\"output_scalars\":" << scalars << ",\"max_abs\":" << max_abs << ",\"sum_abs\":" << sum_abs
              << ",\"repetitions\":" << opt.repetitions << ",\"ns_per_call\":";
    array(samples);
    std::cout << "}\n" << std::flush;
}

std::array<double,3> vec(const double *p) { return {p[0],p[1],p[2]}; }
State state(const double *p) { return {vec(p),vec(p+3)}; }

auto sf_leg(const double *row, std::span<const double> throttles, double tof, double mf)
{
    return kep3::leg::sims_flanagan(state(row),row[6],{throttles.begin(),throttles.end()},
        state(row+7),mf,tof,row[15],row[16],row[17],row[18]);
}

std::vector<double> sf_fit(const double *row, const std::vector<double> &x)
{
    const auto n=x.size()-2;
    const auto leg=sf_leg(row,{x.data(),n},x[n],x[n+1]);
    return flat(-x[n+1],leg.compute_mismatch_constraints(),leg.compute_throttle_constraints());
}

void sf(const Dataset &data, const Options &opt)
{
    const auto n=data.width-19;
    auto make=[&](std::size_t i) {
        const auto r=data.row(i);
        return sf_leg(r,{r+19,n},r[14],r[13]);
    };
    std::vector<kep3::leg::sims_flanagan> legs;
    std::vector<std::vector<double>> chromosomes;
    for (std::size_t i=0; i<data.count; ++i) {
        const auto r=data.row(i);
        legs.push_back(make(i));
        chromosomes.emplace_back(r+19,r+19+n);
        chromosomes.back().insert(chromosomes.back().end(),{r[14],r[13]});
    }
    measure(data,"construct",opt,make,[&](const auto &){return flat(n/3);});
    measure(data,"mismatch",opt,[&](auto i){return legs[i].compute_mismatch_constraints();},
        [](const auto &v){return flat(v);});
    measure(data,"jacobian",opt,[&](auto i){return legs[i].compute_mc_grad();},
        [](const auto &v){return std::apply([](const auto &...q){return flat(q...);},v);});
    measure(data,"udp-analytic",opt,[&](auto i) {
        const auto &x=chromosomes[i];
        const auto leg=make(i);
        const auto [start,end,controls]=leg.compute_mc_grad();
        const auto throttle=leg.compute_tc_grad();
        const auto d=x.size();
        std::vector<double> out(d*(8+n/3),0.);
        out[d-1]=-1.;
        for (std::size_t k=0;k<7;++k) {
            std::copy_n(controls.data()+k*(n+1),n+1,out.data()+(k+1)*d);
            out[(k+2)*d-1]=end[k*7+6];
        }
        for (std::size_t k=0;k<n/3;++k)
            std::copy_n(throttle.data()+k*n,n,out.data()+(8+k)*d);
        return out;
    },[](auto v){return v;});
    measure(data,"udp-numerical",opt,[&](auto i){
        const auto &x=chromosomes[i];
        auto plus=x,minus=x;
        const auto d=x.size(),m=sf_fit(data.row(i),x).size();
        std::vector<double> out(d*m,0.);
        for (std::size_t j=0;j<d;++j) {
            const auto h=std::max(1.,std::abs(x[j]))*1e-8;
            plus[j]+=h;minus[j]-=h;
            const auto u=sf_fit(data.row(i),plus), l=sf_fit(data.row(i),minus);
            for (std::size_t k=0;k<m;++k) out[k*d+j]=(u[k]-l[k])/(2*h);
            plus[j]=x[j];minus[j]=x[j];
        }
        return out;
    },[](auto v){return v;});
}

void zoh(const Dataset &data,const Options &opt,std::size_t n)
{
    auto nominal=kep3::ta::get_ta_zoh_kep(1e-14);
    auto var=kep3::ta::get_ta_zoh_kep_var(opt.derivative_tol);
    nominal.get_pars_data()[4]=var.get_pars_data()[4]=1./3000.;
    std::vector<kep3::leg::zoh> legs;
    legs.reserve(data.count);
    for (std::size_t i=0;i<data.count;++i) {
        const auto r=data.row(i);
        legs.emplace_back(std::vector<double>(r,r+7),std::vector<double>(r+15+n,r+15+5*n),
            std::vector<double>(r+7,r+14),std::vector<double>(r+14,r+15+n),.5,std::make_pair(nominal,var));
    }
    measure(data,"mismatch",opt,[&](auto i){return legs[i].compute_mismatch_constraints();},
        [](const auto &v){return flat(v);});
    measure(data,"jacobian",opt,[&](auto i){return legs[i].compute_mc_grad();},
        [](const auto &v){return std::apply([](const auto &...q){return flat(q...);},v);});
}

int main(int argc,char **argv)
{
    try {
        if (argc<2) throw std::runtime_error("usage: pykep-native-benchmark FIXTURE [--passes=N] [--repetitions=N] [--values=N]");
        Options opt;
        for (int i=2;i<argc;++i) {
            const std::string a=argv[i];
            if (a.starts_with("--passes=")) opt.passes=std::stoull(a.substr(9));
            else if (a.starts_with("--repetitions=")) opt.repetitions=std::stoull(a.substr(14));
            else if (a.starts_with("--values=")) opt.values=std::stoull(a.substr(9));
            else if (a.starts_with("--zoh-derivative-tol=")) opt.derivative_tol=std::stod(a.substr(21));
            else throw std::runtime_error("unknown flag");
        }
        if (!opt.passes || opt.passes>100 || !opt.repetitions || opt.repetitions>100000 || opt.values>1024)
            throw std::runtime_error("invalid sample controls");
        const Dataset data(argv[1]);
        const auto &suite=data.suite;
        std::cout << std::setprecision(17);
        if (suite.starts_with("anomaly") && data.width==2) {
            measure(data,"m2e",opt,[&](auto i){const auto r=data.row(i);return kep3::m2e(r[0],r[1]);},
                [](double e){return flat(std::sin(e),std::cos(e));});
            if (suite.ends_with("-error"))
                measure(data,"m2e-roundtrip",opt,[&](auto i){
                    const auto r=data.row(i);
                    const auto m=kep3::e2m(kep3::m2e(r[0],r[1]),r[1]);
                    return .5*(std::abs(std::sin(m)-std::sin(r[0]))+std::abs(std::cos(m)-std::cos(r[0])));
                },[](double e){return flat(e);});
        } else if ((suite.starts_with("lagrangian") || suite.starts_with("roundtrip") || suite.starts_with("stm"))
            && data.width==7) {
            measure(data,"state",opt,[&](auto i){const auto r=data.row(i);return kep3::propagate_lagrangian(state(r),r[6],1.,false).first;},
                [](const auto &v){return flat(v);});
            if (suite.starts_with("stm"))
                measure(data,"state-stm",opt,[&](auto i){
                    const auto r=data.row(i);return kep3::propagate_lagrangian(state(r),r[6],1.,true);
                },[](const auto &v){return flat(v.first,*v.second);});
            if (suite.starts_with("roundtrip"))
                measure(data,"roundtrip-error",opt,[&](auto i){
                    const auto r=data.row(i);
                    const auto y=kep3::propagate_lagrangian(state(r),r[6],1.,false).first;
                    const auto z=kep3::propagate_lagrangian(y,-r[6],1.,false).first;
                    double err=0.;
                    for (std::size_t j=0;j<6;++j) err+=std::abs(z[j/3][j%3]-r[j]);
                    return err;
                },[](double e){return flat(e);});
        } else if ((suite=="lambert0" || suite=="lambert20") && data.width==9) {
            const auto n=suite=="lambert0" ? 0 : 20;
            measure(data,"solve",opt,[&](auto i){
                const auto r=data.row(i);
                return kep3::lambert_problem(vec(r),vec(r+3),r[6],r[8],r[7]!=0.,n);
            },[](const auto &l){
                std::vector<double> values;
                for (std::size_t i=0;i<l.get_v0().size();++i) {
                    flatten(values,l.get_v0()[i]);flatten(values,l.get_v1()[i]);
                }
                return values;
            });
        } else if (suite.starts_with("sf") && data.width>=22 && (data.width-19)%3==0) {
            sf(data,opt);
        } else if (suite.starts_with("zoh")) {
            const auto n=std::stoul(suite.substr(3));
            if (n<1 || n>512 || data.width!=15+5*n) throw std::runtime_error("invalid ZOH shape");
            zoh(data,opt,n);
        } else throw std::runtime_error("unknown suite or shape");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
