/// @file
/// @brief Input validation of the `peclet.amr` Python binding (python/amr_bindings.cpp only).
///
/// The C++ AMR API (include/peclet/amr/) trusts its caller; the public Python surface refuses a
/// NaN/Inf or an inadmissible value at the binding, as ValueError (std::invalid_argument), before
/// it reaches the octree or the solver. A valid value takes exactly the path it took before. The
/// message names the callable, the parameter, the admissible range with its dimension (amr
/// computes in the caller's consistent units) and the offending value.
///
/// Two kinds of entries: the check helpers (`finite`, `positive`, ...), called from the binding's
/// own constructor lambdas and from the four wrapper methods whose six or seven parameters would
/// make a forwarding wrapper a second long-parameter-list function (`*_args`: refine_to_sphere,
/// adapt, refine_to_gap_floor, Poisson.solve), and the CHECKED METHODS (`setDt<F>`, ...), free
/// functions bound in place of a member pointer: they keep the Python parameter names, defaults
/// and return type (same `__nb_signature__`), and keep the validation branches out of NB_MODULE.
/// The wrapper classes are defined in the binding TU, so the checked methods are templates on them.
/// Binding-only: no header under include/ includes this.
///
/// Ranges are stated only where the docstring, the solver's use or the data structure makes them
/// evident. Where a sign or 0 is a switch the check is finite only: Poisson.solve tol <= 0 disables
/// the early stop, and Octree.adapt's thresholds outside the indicator's [0, 1] mean "never".
/// Unchecked on purpose: DistributedOctree.face_neighbor_gather's `sentinel` (any value, NaN
/// included, is a legitimate marker).
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace peclet::amr::pybind::chk {

namespace nb = nanobind;
using Field = nb::ndarray<double, nb::c_contig>;
using Spacing = std::variant<double, std::array<double, 3>>;
using Sdf = std::function<double(double, double, double)>;
using LevelFn = std::function<unsigned(double, double, double)>;

/// The per-axis Morton code width of the 3D BlockOctree (Bits = 21): a block spans at most 2**21
/// finest cells per axis (block_octree.hpp: brick * 2**lmax <= 2**Bits).
inline constexpr std::int64_t kMaxLevel = 21;
inline constexpr std::int64_t kMaxCells = std::int64_t{1} << kMaxLevel;

inline std::string num(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%g", v);
  return b;
}
inline void require(bool ok, const char* who, const std::string& name, const std::string& rule,
                    double v) {
  if (!ok) {
    throw std::invalid_argument(std::string(who) + ": " + name + " must be " + rule + ", got " +
                                num(v));
  }
}
inline void finite(const char* who, const std::string& name, double v, const char* dim) {
  require(std::isfinite(v), who, name, std::string("finite (") + dim + ")", v);
}
inline void positive(const char* who, const std::string& name, double v, const char* dim) {
  require(std::isfinite(v) && v > 0.0, who, name, std::string("finite and > 0 (") + dim + ")", v);
}
inline void nonneg(const char* who, const std::string& name, double v, const char* dim) {
  require(std::isfinite(v) && v >= 0.0, who, name, std::string("finite and >= 0 (") + dim + ")", v);
}
/// A count (iterations, samples): an integer >= `lo`.
inline void count(const char* who, const std::string& name, std::int64_t v, std::int64_t lo) {
  require(v >= lo, who, name, "an integer >= " + std::to_string(lo), static_cast<double>(v));
}
/// An integer selector or index in [lo, hi].
inline void inRange(const char* who, const std::string& name, std::int64_t v, std::int64_t lo,
                    std::int64_t hi) {
  require(v >= lo && v <= hi, who, name,
          "an integer in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]",
          static_cast<double>(v));
}
/// A tree depth or level: at most the Morton code width.
inline void depth(const char* who, const char* name, unsigned v) {
  inRange(who, name, static_cast<std::int64_t>(v), 0, kMaxLevel);
}
/// A level of this octree: 0 (finest) .. lmax (root).
inline void level(const char* who, const char* name, unsigned v, unsigned lmax) {
  inRange(who, name, static_cast<std::int64_t>(v), 0, static_cast<std::int64_t>(lmax));
}
/// A per-leaf array: length num_leaves.
inline void length(const char* who, const char* name, std::size_t len, std::int64_t n) {
  require(static_cast<std::int64_t>(len) == n, who, std::string(name) + " length",
          "num_leaves = " + std::to_string(n), static_cast<double>(len));
}

/// Per-component forms for a triple: the parameter is named `name[i]`.
inline std::string component(const char* name, std::size_t i) {
  return std::string(name) + "[" + std::to_string(i) + "]";
}
inline void finiteEach(const char* who, const char* name, const std::array<double, 3>& a,
                       const char* dim) {
  for (std::size_t i = 0; i < 3; ++i) {
    finite(who, component(name, i), a[i], dim);
  }
}
inline void positiveEach(const char* who, const char* name, const std::array<double, 3>& a,
                         const char* dim) {
  for (std::size_t i = 0; i < 3; ++i) {
    positive(who, component(name, i), a[i], dim);
  }
}

// --- composite checks of the module functions and the constructor lambdas -------------------

/// predict_hierarchy / spacing(s)_from_extent: the depth (2**lmax must fit the Morton code).
inline void ladder(const char* who, unsigned lmax) {
  depth(who, "lmax", lmax);
}
inline void extentLadder(const char* who, const std::array<double, 3>& extent, unsigned lmax) {
  depth(who, "lmax", lmax);
  positiveEach(who, "extent", extent, "length");
}

/// The placement shared by Octree and DistributedOctree: depth, origin, spacing or extent.
inline void placement(const char* who, unsigned lmax, const std::array<double, 3>& origin,
                      const std::optional<Spacing>& spacing,
                      const std::optional<std::array<double, 3>>& extent) {
  depth(who, "lmax", lmax);
  finiteEach(who, "origin", origin, "length");
  if (spacing && std::holds_alternative<double>(*spacing)) {
    positive(who, "spacing", std::get<double>(*spacing), "length");
  }
  if (spacing && !std::holds_alternative<double>(*spacing)) {
    positiveEach(who, "spacing", std::get<std::array<double, 3>>(*spacing), "length");
  }
  if (extent) {
    positiveEach(who, "extent", *extent, "length");
  }
}
/// Octree: one block, so the finest grid itself must fit the Morton code (<= 2**21 per axis). A
/// DistributedOctree's GLOBAL grid may exceed it (the codes are block-local).
inline void octree(const std::array<long, 3>& cells, unsigned lmax,
                   const std::array<double, 3>& origin, const std::optional<Spacing>& spacing,
                   const std::optional<std::array<double, 3>>& extent) {
  placement("Octree", lmax, origin, spacing, extent);
  for (std::size_t i = 0; i < 3; ++i) {
    require(cells[i] <= kMaxCells, "Octree", component("cells", i),
            "<= 2**21 finest cells per axis (the Morton code width; a count)",
            static_cast<double>(cells[i]));
  }
}

// --- checked methods: Octree / DistributedOctree (bindOctreeCommon) ------------------------

inline void sphereArgs(const std::array<double, 3>& center, double radius, unsigned target_level,
                       double band, unsigned lmax) {
  finiteEach("refine_to_sphere", "center", center, "length");
  positive("refine_to_sphere", "radius", radius, "length");
  level("refine_to_sphere", "target_level", target_level, lmax);
  finite("refine_to_sphere", "band", band, "finest cells");
}
template <class T>
auto refineToSdf(T& t, Sdf sdf, unsigned target_level, double band, bool balance) {
  level("refine_to_sdf", "target_level", target_level, t.lmax());
  finite("refine_to_sdf", "band", band, "finest cells");
  return t.refine_to_sdf(std::move(sdf), target_level, band, balance);
}
template <class T>
auto lohnerIndicator(T& t, const Field& field, double eps) {
  nonneg("lohner_indicator", "eps", eps, "dimensionless");
  return t.lohner_indicator(field, eps);
}
/// adapt: the thresholds are compared with the indicator in [0, 1]; a value outside it means
/// "never refine" / "never coarsen", so finite only.
inline void adaptArgs(double refine_thresh, double coarsen_thresh, unsigned finest_level,
                      unsigned lmax, double eps) {
  finite("adapt", "refine_thresh", refine_thresh, "dimensionless");
  finite("adapt", "coarsen_thresh", coarsen_thresh, "dimensionless");
  level("adapt", "finest_level", finest_level, lmax);
  nonneg("adapt", "eps", eps, "dimensionless");
}

// --- checked methods: Octree only ----------------------------------------------------------

template <class T>
auto find(const T& t, std::array<double, 3> x) {
  finiteEach("find", "x", x, "length");
  return t.find(x);
}
template <class T>
bool refineLeaf(T& t, std::int64_t i) {
  inRange("refine_leaf", "i", i, 0, static_cast<std::int64_t>(t.num_leaves()) - 1);
  return t.refine_leaf(i);
}
template <class T>
auto refineToSdfGraded(T& t, Sdf sdf, LevelFn target_level, double band, bool balance) {
  finite("refine_to_sdf_graded", "band", band, "cells of the level being created");
  return t.refine_to_sdf_graded(std::move(sdf), std::move(target_level), band, balance);
}
inline void gapFloorArgs(unsigned coarsest_level, unsigned lmax, double n, double band) {
  level("refine_to_gap_floor", "coarsest_level", coarsest_level, lmax);
  positive("refine_to_gap_floor", "n", n, "cells per gap, dimensionless");
  finite("refine_to_gap_floor", "band", band, "cells of the level being created");
}

// --- checked methods: DistributedOctree -----------------------------------------------------

/// rebalance: `fields` is (num_leaves, K) with K >= 1 (a 1-D array read a shape it does not have,
/// and K = 0 divided by zero).
template <class D>
auto rebalance(D& d, const Field& fields) {
  require(fields.ndim() == 2, "rebalance", "fields.ndim", "2, a (num_leaves, K) array",
          static_cast<double>(fields.ndim()));
  count("rebalance", "fields.shape[1]", static_cast<std::int64_t>(fields.shape(1)), 1);
  return d.rebalance(fields);
}

// --- checked methods: Poisson ---------------------------------------------------------------

/// Poisson.solve: `tol` <= 0 disables the early stop, so finite only.
inline void solveArgs(int cycles, int pre, int post, double tol) {
  count("Poisson.solve", "cycles", cycles, 0);
  count("Poisson.solve", "pre", pre, 0);
  count("Poisson.solve", "post", post, 0);
  finite("Poisson.solve", "tol", tol, "residual units");
}

// --- checked methods: Flow ------------------------------------------------------------------

inline void flowProperties(double density, double viscosity, double dt) {
  positive("Flow", "density", density, "mass/length^3");
  nonneg("Flow", "viscosity", viscosity, "mass/(length*time)");
  positive("Flow", "dt", dt, "time");
}
/// Both constructors (Octree, DistributedOctree): validate, then construct in place.
template <class F, class O>
void initFlow(F* self, O& octree, double density, double viscosity, double dt) {
  flowProperties(density, viscosity, dt);
  new (self) F(octree, density, viscosity, dt);
}
template <class F>
void setBodyForce(F& f, double fx, double fy, double fz) {
  finiteEach("set_body_force", "f", {fx, fy, fz}, "force/length^3");
  f.set_body_force(fx, fy, fz);
}
/// The closure orders: 2 = wall-anchored quadratic, 1 = linear (core ghost_closure.hpp
/// gpOrderWeights; any other value silently meant 1).
template <class F>
void setGhostProjection(F& f, bool on, int matrix_order, int rhs_order) {
  inRange("set_ghost_projection", "matrix_order", matrix_order, 1, 2);
  inRange("set_ghost_projection", "rhs_order", rhs_order, 1, 2);
  f.set_ghost_projection(on, matrix_order, rhs_order);
}
template <class F>
void setGhostSampled(F& f, bool on, double rho, long max_samples) {
  positive("set_ghost_sampled", "rho", rho, "dimensionless, a factor on the cell width");
  count("set_ghost_sampled", "max_samples", max_samples, 0);
  f.set_ghost_sampled(on, rho, max_samples);
}
/// set_pressure / set_velocity before set_solid (or between begin_adapt and finish_adapt) wrote
/// num_leaves values into unallocated state: refused with the RuntimeError step() raises there.
template <class F>
void setPressure(F& f, const Field& values) {
  length("set_pressure", "values", values.shape(0), static_cast<std::int64_t>(f.num_leaves()));
  f.engine().requireOperator("setPressure");
  f.set_pressure(values);
}
template <class F>
void setDt(F& f, double dt) {
  positive("set_dt", "dt", dt, "time");
  f.set_dt(dt);
}
template <class F>
void setCfScheme(F& f, int scheme) {
  inRange("set_cf_scheme", "scheme", scheme, 0, 1);
  f.set_cf_scheme(scheme);
}
template <class F>
void setAdvectionScheme(F& f, int scheme) {
  inRange("set_advection_scheme", "scheme", scheme, 0, 1);
  f.set_advection_scheme(scheme);
}
/// n < 1 was silently clamped to 1 (AmrFlow::setOuterIterations).
template <class F>
void setOuterIterations(F& f, int n, double tol) {
  count("set_outer_iterations", "n", n, 1);
  nonneg("set_outer_iterations", "tol", tol, "velocity units");
  f.set_outer_iterations(n, tol);
}
template <class F>
void setPressureTolerance(F& f, double rtol) {
  nonneg("set_pressure_tolerance", "rtol", rtol, "dimensionless");
  f.set_pressure_tolerance(rtol);
}
template <class F>
void setMomentumTolerance(F& f, double rtol) {
  nonneg("set_momentum_tolerance", "rtol", rtol, "dimensionless");
  f.set_momentum_tolerance(rtol);
}
template <class F>
void step(F& f, int mom_iters, int pres_iters) {
  count("step", "mom_iters", mom_iters, 0);
  count("step", "pres_iters", pres_iters, 0);
  f.step(mom_iters, pres_iters);
}
template <class F>
void project(F& f, int pres_iters) {
  count("project", "pres_iters", pres_iters, 0);
  f.project(pres_iters);
}
template <class F>
void setVelocity(F& f, int component, const Field& values) {
  inRange("set_velocity", "component", component, 0, 2);
  length("set_velocity", "values", values.shape(0), static_cast<std::int64_t>(f.num_leaves()));
  f.engine().requireOperator("setVelocity");
  f.set_velocity(component, values);
}
/// set_solid_spheres: finite centers and finite positive radii, over every element (the member
/// checks the shapes after this).
using ConstField = nb::ndarray<const double, nb::c_contig>;
inline void spheres(const ConstField& centers, const ConstField& radii) {
  for (std::size_t i = 0; i < centers.size(); ++i) {
    finite("set_solid_spheres", "centers.flat[" + std::to_string(i) + "]", centers.data()[i],
           "length");
  }
  for (std::size_t i = 0; i < radii.size(); ++i) {
    positive("set_solid_spheres", "radii[" + std::to_string(i) + "]", radii.data()[i], "length");
  }
}
template <class F>
void setSolidSpheres(F& f, const ConstField& centers, const ConstField& radii, bool periodic) {
  spheres(centers, radii);
  f.set_solid_spheres(centers, radii, periodic);
}

}  // namespace peclet::amr::pybind::chk
