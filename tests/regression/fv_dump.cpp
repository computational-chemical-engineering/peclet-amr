// Byte-gate driver for the distributed finite-volume operator (peclet::amr::DistributedFvOperator),
// which `peclet.amr` does not expose. Run by tests/regression/state_hash.py under `mpirun -np N`;
// NOT a ctest (the correctness checks are tests/test_amr_distributed_{fv,openness}_mpi.cpp).
//
// The setup is the one of those tests: 4^3 root cells, lmax 2, periodic [0,1)^3, the lower octant
// refined two levels and 2:1-balanced, so 2:1 faces cross block boundaries at np > 1. The operator
// is built twice (openness-free, and with the smooth openness of the openness test), applied to a
// smooth field, and 20 damped-Jacobi sweeps are run. Rank 0 prints every value as a hex float
// (`%a`, exact) in ascending global Morton code, so the output is a decomposition-independent,
// bit-exact text; state_hash.py hashes it.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

#include "peclet/amr/distributed_fv.hpp"
#include "peclet/amr/distributed_octree.hpp"
#include "peclet/core/common/mpi.hpp"

using namespace peclet::core;
using namespace peclet::amr;

namespace {

constexpr unsigned kBits = 21;
using DO = DistributedOctree<3, kBits>;
using M = DO::M;
using Code = DO::Code;

double openFn(const Vec<3>& p, int /*axis*/) {
  const double k = 2.0 * M_PI;
  return 0.6 + 0.35 * std::sin(k * p[0]) * std::cos(k * p[1]) * std::cos(k * p[2]);
}

double fAt(Code gc, double h0) {
  auto o = M::from_code(gc).decode();
  double cx = ((double)o[0] + 0.5) * h0, cy = ((double)o[1] + 0.5) * h0,
         cz = ((double)o[2] + 0.5) * h0;
  const double k = 2.0 * M_PI;
  return std::sin(k * cx) * std::cos(k * cy) + std::cos(k * cz) * std::sin(k * cx);
}

void makeGraded(DO& d) {
  for (int pass = 0; pass < 2; ++pass) {
    d.local().refineIf([&](Code c, unsigned lvl) -> bool {
      if (lvl == 0)
        return false;
      auto o = M::from_code(c).decode();
      for (int dd = 0; dd < 3; ++dd)
        if ((long)o[dd] + d.blockFineOrigin()[dd] >= 8)
          return false;
      return true;
    });
  }
  d.balance();
}

// Gather (code, values...) rows to rank 0 and print them sorted by code.
void dump(const char* tag, const DO& d, const std::vector<std::vector<double>>& cols) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const Index n = d.local().numLeaves();
  const int nc = static_cast<int>(cols.size());
  std::vector<double> rows(static_cast<std::size_t>(n) * (nc + 1));
  for (Index i = 0; i < n; ++i) {
    const auto gc = static_cast<std::uint64_t>(d.globalCode(i));
    double* r = &rows[static_cast<std::size_t>(i) * (nc + 1)];
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::copy_n(reinterpret_cast<const char*>(&gc), sizeof(gc), reinterpret_cast<char*>(r));
    for (int c = 0; c < nc; ++c)
      r[c + 1] = cols[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)];
  }
  int cnt = static_cast<int>(rows.size());
  std::vector<int> cnts(static_cast<std::size_t>(size)), displ(static_cast<std::size_t>(size), 0);
  MPI_Gather(&cnt, 1, MPI_INT, cnts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0)
    std::partial_sum(cnts.begin(), cnts.end() - 1, displ.begin() + 1);
  std::vector<double> all(rank == 0 ? static_cast<std::size_t>(displ.back() + cnts.back()) : 0);
  MPI_Gatherv(rows.data(), cnt, MPI_DOUBLE, all.data(), cnts.data(), displ.data(), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
  if (rank != 0)
    return;
  const std::size_t w = static_cast<std::size_t>(nc + 1), nr = all.size() / w;
  std::vector<std::uint64_t> code(nr);
  for (std::size_t r = 0; r < nr; ++r)
    std::copy_n(reinterpret_cast<const char*>(&all[r * w]), sizeof(std::uint64_t),
                reinterpret_cast<char*>(&code[r]));
  std::vector<std::size_t> ord(nr);
  std::iota(ord.begin(), ord.end(), std::size_t(0));
  std::sort(ord.begin(), ord.end(),
            [&](std::size_t a, std::size_t b) { return code[a] < code[b]; });
  std::printf("%s %zu\n", tag, nr);
  for (std::size_t r : ord) {
    std::printf("%llx", static_cast<unsigned long long>(code[r]));
    for (std::size_t c = 1; c < w; ++c)
      std::printf(" %a", all[r * w + c]);
    std::printf("\n");
  }
}

template <class InitFn>
void runCase(const char* tag, InitFn&& initOp) {
  const long Nr = 4;
  const unsigned lmax = 2;
  const double h0 = 1.0 / (Nr * (1 << lmax));
  AmrGeometry<3> geo;
  geo.setIsotropic(h0);
  const std::array<bool, 3> per{true, true, true};
  DO d;
  d.init(IVec<3>{Nr, Nr, Nr}, lmax, geo, per, MPI_COMM_WORLD);
  makeGraded(d);
  DistributedFvOperator<3, kBits> op;
  initOp(op, d);
  const Index n = d.local().numLeaves();
  std::vector<double> f(static_cast<std::size_t>(n)), lu, x(static_cast<std::size_t>(n), 0.0);
  for (Index i = 0; i < n; ++i)
    f[static_cast<std::size_t>(i)] = fAt(d.globalCode(i), h0);
  op.apply(f, lu);
  op.jacobi(x, f, 20, 0.8);
  const double res = op.residualNorm(x, f);
  dump(tag, d, {f, lu, x});
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    std::printf("%s residual %a\n", tag, res);
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  runCase("fv_plain", [](DistributedFvOperator<3, kBits>& op, DO& d) { op.init(d); });
  runCase("fv_open", [](DistributedFvOperator<3, kBits>& op, DO& d) { op.init(d, openFn); });
  MPI_Finalize();
  return 0;
}
