// amr — the EXACT coarsest-level solve of the pressure multigrid (docs/amr_mg_depth.md §6.6).
//
// The §6.2 ladder lifts the root while the grid halves into a cube level. On a badly factored root
// grid it runs out early — 100^3 stops at 25^3, a 64x64x4 slab stops at 32x32x2 — and 60 damped
// Jacobi sweeps are then not a solve but a smoother (the §6.6 amplification table: E=4 gives 8e-9
// in 60 sweeps, E=8 gives 8e-3, E=25 gives 0.60). A V-cycle whose coarsest level is not solved is
// not mesh-independent, so the bottom there is an AGGLOMERATED direct-ish solve: the level's own FV
// operator assembled as a sparse SPD matrix and solved by GraphAMG-preconditioned CG. This is
// flow's `PECLET_FLOW_AGGLOM_EXTENT` bottom, ported onto core's `peclet::core::solver::GraphAMG` —
// the same class flow uses.
//
// WHERE IT RUNS (§5.5). The SETUP is always host — smoothed aggregation is a sequential greedy
// pass and runs once per `build()`. The per-V-cycle SOLVE follows the bottom's size: at most
// `kHostMax` = 10^4 rows it stages n_b doubles each way and runs the CG on the host (the round trip
// is cheaper than the kernel launches at that size); above it the whole CG runs on the device
// against core's `GraphAMGDevice`, so a big elongated-brick bottom (§5.5: `512x512x4` lifts to
// `256x256x2` = 131 k cells) never leaves the device. The two paths are the same algorithm; only
// the dot products reassociate, which perturbs an inner solve converged to `kTol` by ~1e-12.
// The threshold is §5.5's; §11.11 later MEASURED it to be about an order of magnitude too high
// (the host CG is ~16 % faster than the 60 sweeps at a 64-cell bottom, ~28 % slower at 4096 rows)
// and left it, with those numbers, for whoever builds the device path out.
//
// It is used by `Multigrid` single-rank and therefore by the replicated stage (§6.5), which IS a
// single-rank Multigrid on the gathered coarsest in-place level.
//
// THE SIGN. §6.6 delegates the sign and volume convention to the implementer, to be confirmed
// against fv_op.hpp rather than read off the note. The FV operator is
//     (L u)_i = invVol_i · ( Σ_f w_f (u_j − u_i) − bcDiag_i u_i ),
// so with S_ii = Σ_f w_f + bcDiag_i and S_ij = −w_f (the note's S, symmetric and positive
// semi-definite):
//     (S u)_i = −(1/invVol_i) · (L u)_i,
// and the bottom problem L x = b becomes S x = −b/invVol. `Multigrid::setBottomCheck` /
// `Multigrid::bottomResidual` is the gate that this is right: after every bottom solve it
// recomputes |b − L x| with the V-CYCLE's own FvOp (§6.6 step 6), which is a different code path
// from this class's CSR and would not agree if the sign or the volume factor were wrong.
#ifndef PECLET_AMR_AMG_BOTTOM_HPP
#define PECLET_AMR_AMG_BOTTOM_HPP

#include <algorithm>
#include <cmath>
#include <vector>

#include "peclet/amr/common.hpp"
#include "peclet/amr/poisson.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/solver/graph_amg.hpp"
#include "peclet/core/solver/graph_amg_device.hpp"

namespace peclet::amr {

/// The agglomerated exact bottom of the pressure multigrid (docs/amr_mg_depth.md §6.6): the
/// coarsest level's FV Laplacian assembled as the SPD matrix S = −Vol·L, with identity rows for
/// fully closed cells and the constant nullspace projected per connected fluid component, solved
/// by `GraphAMG`-preconditioned CG to relative `kTol` = 1e-10 (cap `kMaxIters` = 100).
///
/// Owned by `Multigrid` (single-rank, and therefore the replicated stage's continued ladder) and
/// by `DistributedFlowMultigrid` at np = 1; selected by `Multigrid::setBottom` —
/// `Flow.diagnostics.set_pressure_bottom('auto' | 'smoother' | 'agglomerated')`. It changes the
/// preconditioner, never the converged pressure. LOCAL: no MPI anywhere in the class — a
/// replicated tail runs one identical copy per rank. Setup is host, once per `build()`; the
/// per-V-cycle solve is host or device by size (`kHostMax`). Throws nothing of its own.
template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class AmgBottom {
 public:
  using Poisson = AmrPoisson<Dim, Bits>;  ///< the level operator the matrix is assembled from

  /// Above this many rows the per-V-cycle solve runs on the device (§5.5).
  static constexpr Index kHostMax = 10000;

  /// Assemble S from the coarsest level's AmrPoisson and build the GraphAMG hierarchy.
  /// `singular` = the operator has a constant nullspace per connected fluid component (the
  /// periodic pure-Neumann pressure, which is every AmrFlow path).
  void build(const Poisson& ap, bool singular) {
    const auto A = ap.assembleFv();
    n_ = static_cast<Index>(A.invVol.size());
    singular_ = singular;
    S_ = core::solver::HostCsrOp{};
    S_.n = n_;
    S_.start = A.start;
    S_.nbr = A.nbr;
    S_.coef.assign(A.coef.size(), 0.0);
    S_.diag.assign(static_cast<std::size_t>(n_), 0.0);
    identity_.assign(static_cast<std::size_t>(n_), 0);
    nIdentity_ = 0;
    for (Index i = 0; i < n_; ++i) {
      double s = 0.0;
      for (Index k = A.start[static_cast<std::size_t>(i)];
           k < A.start[static_cast<std::size_t>(i) + 1]; ++k) {
        S_.coef[static_cast<std::size_t>(k)] = -A.coef[static_cast<std::size_t>(k)];
        s += A.coef[static_cast<std::size_t>(k)];  // resum: S·1 = 0 exactly in the singular case
      }
      S_.diag[static_cast<std::size_t>(i)] = s + A.bcDiag[static_cast<std::size_t>(i)];
      if (S_.diag[static_cast<std::size_t>(i)] <= 0.0) {
        // A fully closed cell (every face α = 0): an IDENTITY row, rhs 0 and solution 0. Its whole
        // row and column are already zero, so setting the diagonal to 1 isolates it without
        // touching any other row — and keeps the matrix SPD for the AMG setup.
        identity_[static_cast<std::size_t>(i)] = 1;
        S_.diag[static_cast<std::size_t>(i)] = 1.0;
        ++nIdentity_;
      }
    }
    buildComponents(A);
    const core::solver::AmgParams prm;  // defaults (§6.6 step 4)
    onDevice_ = (n_ > kHostMax);
    dIn_ = View<double>("amgb_in", static_cast<std::size_t>(n_));
    dOut_ = View<double>("amgb_out", static_cast<std::size_t>(n_));
    if (onDevice_) {
      amgD_.build(S_, prm);
      uploadDevice(A);
    } else {
      amg_.build(S_, prm);
      invVol_ = A.invVol;
      r_.assign(static_cast<std::size_t>(n_), 0.0);
      z_.assign(static_cast<std::size_t>(n_), 0.0);
      p_.assign(static_cast<std::size_t>(n_), 0.0);
      q_.assign(static_cast<std::size_t>(n_), 0.0);
      rhs_.assign(static_cast<std::size_t>(n_), 0.0);
      bHost_.assign(static_cast<std::size_t>(n_), 0.0);
      xHost_.assign(static_cast<std::size_t>(n_), 0.0);
      bMirror_ = Kokkos::View<double*, Kokkos::HostSpace>("amgb_b", static_cast<std::size_t>(n_));
      xMirror_ = Kokkos::View<double*, Kokkos::HostSpace>("amgb_x", static_cast<std::size_t>(n_));
    }
  }

  /// True once `build()` has assembled a non-empty level.
  bool ready() const { return n_ > 0; }
  /// Whether the per-V-cycle solve runs on the device (`size() > kHostMax`).
  bool onDevice() const { return onDevice_; }
  /// Rows of the bottom matrix = cells of the coarsest level.
  Index size() const { return n_; }
  /// Fully closed cells (every face openness 0), solved as identity rows with solution 0.
  Index numIdentityRows() const { return nIdentity_; }
  /// Connected fluid components; each carries its own constant nullspace mode.
  int numComponents() const { return nComp_; }
  /// CG iterations of the last `solve` (0 for a zero right-hand side).
  int lastIters() const { return lastIters_; }
  /// ‖rhs − S x‖∞ / ‖rhs‖∞ of the last solve, measured on the CSR the bottom itself assembled.
  /// The V-cycle's own FvOp check (the §6.6 consistency gate) lives in Multigrid::vcycle and is the
  /// one §10 quotes.
  double lastRelResidual() const { return lastRel_; }

  /// x ← the solution of L x = b on the coarsest level, i.e. S x = −b/invVol, projected onto the
  /// complement of the per-component nullspace. Device views, which may be LONGER than `size()`
  /// (the distributed levels carry a ghost tail); only the first `size()` rows are read and
  /// written, and everything runs through this class's own scratch so a subview is never needed.
  void solve(View<const double> b, View<double> x) const {
    auto in = dIn_;
    Kokkos::parallel_for("amr::amgb_in", n_, KOKKOS_LAMBDA(const Index i) { in(i) = b(i); });
    if (onDevice_)
      solveDevice();
    else
      solveHost();
    auto out = dOut_;
    Kokkos::parallel_for("amr::amgb_out", n_, KOKKOS_LAMBDA(const Index i) { x(i) = out(i); });
  }

 private:
  // ---- host path -----------------------------------------------------------------------------
  void solveHost() const {
    Kokkos::deep_copy(bMirror_, dIn_);
    for (Index i = 0; i < n_; ++i)
      bHost_[static_cast<std::size_t>(i)] = bMirror_(i);
    std::fill(xHost_.begin(), xHost_.end(), 0.0);
    for (Index i = 0; i < n_; ++i)
      rhs_[static_cast<std::size_t>(i)] =
          identity_[static_cast<std::size_t>(i)]
              ? 0.0
              : -bHost_[static_cast<std::size_t>(i)] / invVol_[static_cast<std::size_t>(i)];
    project(rhs_);
    r_ = rhs_;
    double bnorm = 0.0;
    for (Index i = 0; i < n_; ++i)
      bnorm = std::max(bnorm, std::fabs(rhs_[static_cast<std::size_t>(i)]));
    lastIters_ = 0;
    lastRel_ = 0.0;
    if (bnorm > 0.0) {
      const double rnorm0 = norm2(r_);
      double rz = 0.0, rzOld = 0.0;
      for (int it = 0; it < kMaxIters; ++it) {
        amg_.apply(r_, z_);
        project(z_);  // flow lesson 1: project per COMPONENT, never the all-cell mean
        rz = dot(r_, z_);
        if (it == 0) {
          p_ = z_;
        } else {
          const double beta = rz / rzOld;
          for (Index i = 0; i < n_; ++i)
            p_[static_cast<std::size_t>(i)] =
                z_[static_cast<std::size_t>(i)] + beta * p_[static_cast<std::size_t>(i)];
        }
        S_.apply(p_, q_);
        const double pq = dot(p_, q_);
        if (!(std::fabs(pq) > 0.0))
          break;
        const double alpha = rz / pq;
        for (Index i = 0; i < n_; ++i) {
          xHost_[static_cast<std::size_t>(i)] += alpha * p_[static_cast<std::size_t>(i)];
          r_[static_cast<std::size_t>(i)] -= alpha * q_[static_cast<std::size_t>(i)];
        }
        rzOld = rz;
        lastIters_ = it + 1;
        if (norm2(r_) <= kTol * rnorm0)
          break;
      }
      project(xHost_);
      for (Index i = 0; i < n_; ++i)
        if (identity_[static_cast<std::size_t>(i)])
          xHost_[static_cast<std::size_t>(i)] = 0.0;
      S_.apply(xHost_, q_);
      double e = 0.0;
      for (Index i = 0; i < n_; ++i)
        e = std::max(
            e, std::fabs(rhs_[static_cast<std::size_t>(i)] - q_[static_cast<std::size_t>(i)]));
      lastRel_ = e / bnorm;
    }
    for (Index i = 0; i < n_; ++i)
      xMirror_(i) = xHost_[static_cast<std::size_t>(i)];
    Kokkos::deep_copy(dOut_, xMirror_);
  }

  static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
      s += a[i] * b[i];
    return s;
  }
  static double norm2(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }

  /// Subtract, per connected component, the mean over that component (a no-op when the operator
  /// is not singular). Identity rows are left alone.
  void project(std::vector<double>& v) const {
    if (!singular_ || nComp_ == 0)
      return;
    std::vector<double> sum(static_cast<std::size_t>(nComp_), 0.0);
    for (Index i = 0; i < n_; ++i) {
      const Index c = comp_[static_cast<std::size_t>(i)];
      if (c >= 0)
        sum[static_cast<std::size_t>(c)] += v[static_cast<std::size_t>(i)];
    }
    for (int c = 0; c < nComp_; ++c)
      sum[static_cast<std::size_t>(c)] /=
          static_cast<double>(compSize_[static_cast<std::size_t>(c)]);
    for (Index i = 0; i < n_; ++i) {
      const Index c = comp_[static_cast<std::size_t>(i)];
      if (c >= 0)
        v[static_cast<std::size_t>(i)] -= sum[static_cast<std::size_t>(c)];
    }
  }

  // ---- device path ---------------------------------------------------------------------------
  void uploadDevice(const typename Poisson::FvAssembled& A) {
    dDiag_ = toDevice(S_.diag, "amgb_diag");
    dStart_ = toDevice(S_.start, "amgb_start");
    dNbr_ = toDevice(S_.nbr, "amgb_nbr");
    dCoef_ = toDevice(S_.coef, "amgb_coef");
    dInvVol_ = toDevice(A.invVol, "amgb_invvol");
    dComp_ = toDevice(comp_, "amgb_comp");
    std::vector<double> ident(static_cast<std::size_t>(n_), 0.0);
    for (Index i = 0; i < n_; ++i)
      ident[static_cast<std::size_t>(i)] = identity_[static_cast<std::size_t>(i)] ? 1.0 : 0.0;
    dIdent_ = toDevice(ident, "amgb_ident");
    const std::size_t nn = static_cast<std::size_t>(n_);
    dR_ = View<double>("amgb_r", nn);
    dZ_ = View<double>("amgb_z", nn);
    dP_ = View<double>("amgb_p", nn);
    dQ_ = View<double>("amgb_q", nn);
    dRhs_ = View<double>("amgb_rhs", nn);
  }

  /// Subtract the per-component mean on the device. The component count is 1 on a connected pore
  /// space and a handful with sealed pockets, so one reduction per component is the cheap shape.
  void projectDevice(View<double> v) const {
    if (!singular_ || nComp_ == 0)
      return;
    auto comp = dComp_;
    for (int c = 0; c < nComp_; ++c) {
      double s = 0.0;
      Kokkos::parallel_reduce(
          "amr::amgb_proj_sum", n_,
          KOKKOS_LAMBDA(const Index i, double& acc) {
            if (comp(i) == c)
              acc += v(i);
          },
          s);
      const double m = s / static_cast<double>(compSize_[static_cast<std::size_t>(c)]);
      Kokkos::parallel_for(
          "amr::amgb_proj_sub", n_, KOKKOS_LAMBDA(const Index i) {
            if (comp(i) == c)
              v(i) -= m;
          });
    }
  }

  void applyS(View<const double> in, View<double> out) const {
    auto diag = dDiag_;
    auto start = dStart_;
    auto nbr = dNbr_;
    auto coef = dCoef_;
    Kokkos::parallel_for(
        "amr::amgb_spmv", n_, KOKKOS_LAMBDA(const Index i) {
          double s = diag(i) * in(i);
          for (Index k = start(i); k < start(i + 1); ++k)
            s += coef(k) * in(nbr(k));
          out(i) = s;
        });
  }

  static double dotDevice(Index n, View<const double> a, View<const double> b) {
    double s = 0.0;
    Kokkos::parallel_reduce(
        "amr::amgb_dot", n, KOKKOS_LAMBDA(const Index i, double& acc) { acc += a(i) * b(i); }, s);
    return s;
  }
  static double maxAbs(Index n, View<const double> a) {
    double m = 0.0;
    Kokkos::parallel_reduce(
        "amr::amgb_maxabs", n,
        KOKKOS_LAMBDA(const Index i, double& acc) {
          const double v = a(i) < 0.0 ? -a(i) : a(i);
          if (v > acc)
            acc = v;
        },
        Kokkos::Max<double>(m));
    return m;
  }

  void solveDevice() const {
    auto rhs = dRhs_;
    auto invVol = dInvVol_;
    auto ident = dIdent_;
    auto b = dIn_;
    View<double> x = dOut_;
    Kokkos::parallel_for(
        "amr::amgb_rhs", n_,
        KOKKOS_LAMBDA(const Index i) { rhs(i) = (ident(i) != 0.0) ? 0.0 : -b(i) / invVol(i); });
    projectDevice(dRhs_);
    Kokkos::deep_copy(x, 0.0);
    Kokkos::deep_copy(dR_, dRhs_);
    const double bnorm = maxAbs(n_, View<const double>(dRhs_));
    lastIters_ = 0;
    lastRel_ = 0.0;
    if (bnorm > 0.0) {
      const double rnorm0 =
          std::sqrt(dotDevice(n_, View<const double>(dR_), View<const double>(dR_)));
      double rz = 0.0, rzOld = 0.0;
      for (int it = 0; it < kMaxIters; ++it) {
        amgD_.apply(dR_, dZ_);
        projectDevice(dZ_);
        rz = dotDevice(n_, View<const double>(dR_), View<const double>(dZ_));
        if (it == 0) {
          Kokkos::deep_copy(dP_, dZ_);
        } else {
          const double beta = rz / rzOld;
          auto p = dP_;
          auto z = dZ_;
          Kokkos::parallel_for(
              "amr::amgb_pupd", n_, KOKKOS_LAMBDA(const Index i) { p(i) = z(i) + beta * p(i); });
        }
        applyS(View<const double>(dP_), dQ_);
        const double pq = dotDevice(n_, View<const double>(dP_), View<const double>(dQ_));
        if (!(std::fabs(pq) > 0.0))
          break;
        const double alpha = rz / pq;
        auto p = dP_;
        auto q = dQ_;
        auto r = dR_;
        Kokkos::parallel_for(
            "amr::amgb_xupd", n_, KOKKOS_LAMBDA(const Index i) {
              x(i) += alpha * p(i);
              r(i) -= alpha * q(i);
            });
        rzOld = rz;
        lastIters_ = it + 1;
        if (std::sqrt(dotDevice(n_, View<const double>(dR_), View<const double>(dR_))) <=
            kTol * rnorm0)
          break;
      }
      projectDevice(x);
      Kokkos::parallel_for(
          "amr::amgb_ident0", n_, KOKKOS_LAMBDA(const Index i) {
            if (ident(i) != 0.0)
              x(i) = 0.0;
          });
      applyS(View<const double>(x), dQ_);
      auto q = dQ_;
      auto rr = dRhs_;
      double e = 0.0;
      Kokkos::parallel_reduce(
          "amr::amgb_resid", n_,
          KOKKOS_LAMBDA(const Index i, double& acc) {
            const double v = rr(i) - q(i);
            const double av = v < 0.0 ? -v : v;
            if (av > acc)
              acc = av;
          },
          Kokkos::Max<double>(e));
      lastRel_ = e / bnorm;
    }
  }

  // ---- shared setup --------------------------------------------------------------------------
  /// Connected components of the fluid graph: union-find over the faces with a POSITIVE
  /// conductance, identity rows excluded. The nullspace of a singular S is the constant on each
  /// component separately — projecting the all-cell mean instead leaves part of the null component
  /// alive and stalls the inner CG (flow's measured lesson, §6.6 step 3).
  void buildComponents(const typename Poisson::FvAssembled& A) {
    parent_.resize(static_cast<std::size_t>(n_));
    for (Index i = 0; i < n_; ++i)
      parent_[static_cast<std::size_t>(i)] = i;
    for (Index i = 0; i < n_; ++i) {
      if (identity_[static_cast<std::size_t>(i)])
        continue;
      for (Index k = A.start[static_cast<std::size_t>(i)];
           k < A.start[static_cast<std::size_t>(i) + 1]; ++k) {
        if (!(A.coef[static_cast<std::size_t>(k)] > 0.0))
          continue;
        const Index j = A.nbr[static_cast<std::size_t>(k)];
        if (j < 0 || j >= n_ || identity_[static_cast<std::size_t>(j)])
          continue;
        unite(i, j);
      }
    }
    comp_.assign(static_cast<std::size_t>(n_), -1);
    nComp_ = 0;
    std::vector<Index> label(static_cast<std::size_t>(n_), -1);
    for (Index i = 0; i < n_; ++i) {
      if (identity_[static_cast<std::size_t>(i)])
        continue;
      const Index r = find(i);
      if (label[static_cast<std::size_t>(r)] < 0)
        label[static_cast<std::size_t>(r)] = nComp_++;
      comp_[static_cast<std::size_t>(i)] = label[static_cast<std::size_t>(r)];
    }
    compSize_.assign(static_cast<std::size_t>(nComp_), 0);
    for (Index i = 0; i < n_; ++i)
      if (comp_[static_cast<std::size_t>(i)] >= 0)
        ++compSize_[static_cast<std::size_t>(comp_[static_cast<std::size_t>(i)])];
  }

  Index find(Index i) const {
    while (parent_[static_cast<std::size_t>(i)] != i) {
      parent_[static_cast<std::size_t>(i)] =
          parent_[static_cast<std::size_t>(parent_[static_cast<std::size_t>(i)])];
      i = parent_[static_cast<std::size_t>(i)];
    }
    return i;
  }
  void unite(Index a, Index b) {
    const Index ra = find(a), rb = find(b);
    if (ra != rb)
      parent_[static_cast<std::size_t>(ra)] = rb;
  }

  /// The inner relative tolerance. §6.6 step 4 says 1e-8 and §6.6 step 6's consistency gate says
  /// max|b − Lx| / max|b| <= 1e-9 — and those two are not jointly satisfiable, because the gate is
  /// a MAX-norm relative residual while the CG stops on a 2-norm one: measured on the §9 cut-cell
  /// pocket case, 1e-8 leaves 2.56e-9 and misses the gate by 2.6x. §11.8's own default is to
  /// tighten the inner tolerance, and 1e-10 costs nothing outside the bottom — the two consistency
  /// numbers become 2.8e-11 and 7.7e-11, the inner CG goes from 7 to 8-10 iterations, and every
  /// outer PCG count in tests/test_amr_mg_bottom.cpp is UNCHANGED (12 / 14 / 16 / 17), exactly the
  /// insensitivity flow measured between 1e-5 and 1e-8.
  static constexpr double kTol = 1e-10;
  static constexpr int kMaxIters = 100;

  Index n_ = 0, nIdentity_ = 0;
  bool singular_ = true, onDevice_ = false;
  int nComp_ = 0;
  core::solver::HostCsrOp S_;
  core::solver::GraphAMG amg_;
  core::solver::GraphAMGDevice amgD_;
  std::vector<double> invVol_;
  std::vector<char> identity_;
  mutable std::vector<Index> parent_;
  std::vector<Index> comp_, compSize_;
  mutable std::vector<double> r_, z_, p_, q_, rhs_, bHost_, xHost_;
  mutable Kokkos::View<double*, Kokkos::HostSpace> bMirror_, xMirror_;
  View<double> dDiag_, dCoef_, dInvVol_, dIdent_;
  View<Index> dStart_, dNbr_, dComp_;
  mutable View<double> dR_, dZ_, dP_, dQ_, dRhs_, dIn_, dOut_;
  mutable int lastIters_ = 0;
  mutable double lastRel_ = 0.0;
};

}  // namespace peclet::amr

#endif  // PECLET_AMR_AMG_BOTTOM_HPP
