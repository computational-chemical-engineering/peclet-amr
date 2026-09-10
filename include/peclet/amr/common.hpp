// peclet-amr — the package namespace and the vocabulary it inherits from peclet-core.
//
// The AMR tree was carved out of peclet::core on 2026-09-10 (suite/docs/QUALITY_PLAN.md D6 / G.2).
// Its sources use core's names — Index, Real, Vec, IVec, View, toDevice, geom::, halo::, decomp::,
// solver:: — unqualified, as they did when they lived inside that namespace. One using-directive
// keeps that vocabulary instead of ~2000 qualifications; a name declared in peclet::amr still hides
// the core one for unqualified lookup, exactly as the nested namespace did.
#ifndef PECLET_AMR_COMMON_HPP
#define PECLET_AMR_COMMON_HPP

#include "peclet/core/common/types.hpp"

namespace peclet::amr {
using namespace peclet::core;  // NOLINT(google-build-using-namespace)
}  // namespace peclet::amr

#endif  // PECLET_AMR_COMMON_HPP
