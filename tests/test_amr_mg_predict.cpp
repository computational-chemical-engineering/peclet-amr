// The pressure-multigrid ladder PREDICTION (docs/amr_mg_depth.md §6.7, ROADMAP C1, WO2).
//
// `predictPressureLadder` is a pure host function: given the global root grid, the tree depth and
// the rank count it re-runs the §6.2 ladder rule on the ORB the DistributedOctree would build, and
// says how many levels there will be, which are in place and which are tail, and what solves the
// bottom. It exists so tests assert against the design rather than against a literal that goes
// stale. Validates:
//   (1) the prediction reproduces every worked ladder of §6.2's table, single-rank AND distributed;
//   (2) on every single-rank row it equals the ladder AmrMultigrid::build actually builds — level
//       count and, on a uniform mesh, per-level cell count.
//
#include <string>
#include <vector>

#include "peclet/amr/block_octree.hpp"
#include "peclet/amr/mg_predict.hpp"
#include "peclet/amr/poisson.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;

// One row of §6.2's worked-ladder table.
struct Row {
  const char* what;
  IVec<3> G;           // the global ROOT grid
  unsigned lmax;       // octree coarsenings the mesh supports (refined to level 0)
  int np;              // ranks
  int levels;          // total predicted levels (in place + tail)
  int inPlace;         // levels before the tail
  const char* bottom;  // "jacobi" | "amg", + "+tail"
  bool buildable;      // single-rank rows we also BUILD and compare against
};

const Row kRows[] = {
    {"uniform 64^3, lmax=0, np=1", {64, 64, 64}, 0, 1, 5, 5, "jacobi", true},
    {"uniform 32^3, lmax=0, np=1", {32, 32, 32}, 0, 1, 4, 4, "jacobi", true},
    {"graded 64^3, lmax=3 (root 8^3)", {8, 8, 8}, 3, 1, 5, 5, "jacobi", true},
    {"byte gate 32^3, lmax=1, np=2", {16, 16, 16}, 1, 2, 4, 4, "jacobi", false},
    {"byte gate 32^3, lmax=1, np=1", {16, 16, 16}, 1, 1, 4, 4, "jacobi", true},
    {"16^3 root, np=8", {16, 16, 16}, 0, 8, 3, 3, "jacobi", false},
    {"12^3 root, np=4 (tail)", {12, 12, 12}, 0, 4, 4, 2, "jacobi+tail", false},
    {"10^3 root, np=1", {10, 10, 10}, 0, 1, 2, 2, "amg", true},
    {"384^3, lmax=0, np=1536 (tail)", {384, 384, 384}, 0, 1536, 9, 4, "jacobi+tail", false},
    {"64x64x4 root", {64, 64, 4}, 0, 1, 2, 2, "amg", true},
};

// The mesh a row describes: a root brick of `G` refined everywhere down to level 0.
BO uniformMeshOf(const IVec<3>& G, unsigned lmax) {
  BO t(G, lmax);
  for (unsigned r = 0; r < lmax; ++r)
    t.refineIf([](Code, unsigned L) { return L > 0; });
  return t;
}

void run() {
  for (const Row& r : kRows) {
    const auto p = predictPressureLadder<3>(r.G, r.lmax, r.np);
    if ((int)p.levels.size() != r.levels || (int)p.numInPlace() != r.inPlace ||
        p.bottomName() != std::string(r.bottom))
      std::fprintf(stderr, "row '%s': levels %d (want %d), in place %d (want %d), bottom %s\n",
                   r.what, (int)p.levels.size(), r.levels, (int)p.numInPlace(), r.inPlace,
                   p.bottomName().c_str());
    PECLET_AMR_CHECK_EQ((long long)p.levels.size(), (long long)r.levels);
    PECLET_AMR_CHECK_EQ((long long)p.numInPlace(), (long long)r.inPlace);
    PECLET_AMR_CHECK(p.bottomName() == std::string(r.bottom));
    PECLET_AMR_CHECK_EQ(p.tail, (p.bottomName().find("+tail") != std::string::npos));

    if (!r.buildable)
      continue;
    // The prediction IS the built ladder: same level count, same per-level cell count.
    AmrMultigrid<3, kBits> mg;
    mg.build(uniformMeshOf(r.G, r.lmax), 1.0);
    PECLET_AMR_CHECK_EQ((long long)mg.numLevels(), (long long)p.levels.size());
    for (std::size_t L = 0; L < mg.numLevels() && L < p.levels.size(); ++L)
      PECLET_AMR_CHECK_EQ((long long)mg.op(L).octree().numLeaves(), p.levels[L].cells);
  }

  // The predicted extents halve, level by level, and every level below the root is 'lifted'.
  const auto p = predictPressureLadder<3>(IVec<3>{64, 64, 64}, 0, 1);
  PECLET_AMR_CHECK(p.levels[0].kind == MgLevelKind::Octree);
  for (std::size_t L = 1; L < p.levels.size(); ++L) {
    PECLET_AMR_CHECK(p.levels[L].kind == MgLevelKind::Lifted);
    PECLET_AMR_CHECK_EQ((long long)p.levels[L].extent[0],
                        (long long)(p.levels[L - 1].extent[0] / 2));
  }
  // liftRoot = false is the pre-C1 ladder, which the prediction deliberately does NOT describe.
  AmrMultigrid<3, kBits> off;
  off.build(uniformMeshOf(IVec<3>{64, 64, 64}, 0), 1.0, /*liftRoot=*/false);
  PECLET_AMR_CHECK_EQ((long long)off.numLevels(), 1LL);
}

}  // namespace

int main() {
  run();
  PECLET_AMR_RETURN_TEST_RESULT();
}
