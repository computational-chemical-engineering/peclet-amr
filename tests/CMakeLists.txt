# Every test is registered through peclet_amr_add_test / peclet_amr_add_mpi_test
# (cmake/PecletAmrTest.cmake): SKIP_RETURN_CODE 77 and the mpi / np8 / bench labels live there.
# Every binary links peclet::amr (the headers + core + morton + Kokkos + MPI).
# The ctest NAMES are the ones the battery had in peclet-core (relocated 2026-09-10), so the
# per-test history in that repo's logs still applies.

function(_amr_test_exe target)
  add_executable(${target} ${target}.cpp)
  target_link_libraries(${target} PRIVATE peclet::amr)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

# ---- single-rank, host octree + serial oracle -----------------------------------------------------
# block_octree       per-block adaptive octree vs the tests/oracle/morton_octree.hpp std::map oracle
# amr_vtu            leaf fields + VTU (UnstructuredGrid) output
# amr_sdf            SDF-driven refinement (refineToSdf) over the shared peclet::core::geom SDF
# amr_poisson        cell-centred FV Poisson on the octree + geometric multigrid
# amr_transfer       restriction / prolongation transfers
# amr_adapt          solution-adaptive refinement (Löhner indicator, conservative remap)
# amr_adapt_transport  adaptivity under explicit scalar transport
# amr_cf_quadratic   quadratic coarse-fine flux + refluxing (2nd-order C/F interface)
# amr_openness       cut-cell openness folded into operator + MG coarsening
# amr_cut_cell       cut-cell Dirichlet sub-cell BC (the xi-polynomial scheme) + kappa
# amr_flow           collocated Stokes momentum+pressure step (host oracle)
# amr_drag           Stokes sphere-array drag vs Zick & Homsy: convergence trend
# amr_face_field     ABC/Basilisk divergence-free face field (uniform + graded 2:1)
# amr_barnes_hut     Barnes-Hut particle-tree mode over the octree
# amr_transport      explicit FV scalar transport (advection-diffusion) on the octree
# amr_cf_vector      C/F interface schemes (cf_scheme.hpp): Martin–Cartwright quadratic deltas
foreach(t IN ITEMS block_octree amr_vtu amr_sdf amr_poisson amr_transfer amr_adapt amr_adapt_transport
                   amr_cf_quadratic amr_openness amr_cut_cell amr_flow amr_drag amr_face_field
                   amr_barnes_hut amr_transport amr_cf_vector)
  _amr_test_exe(test_${t})
  peclet_amr_add_test(${t} test_${t})
endforeach()

# ---- single-rank, device (Kokkos) solver stack ----------------------------------------------------
# block_octree_view  device locate / face-neighbour queries vs the host BlockOctree
# amr_fv_op          device FV operator
# amr_assembly       device operator assembly
# amr_momentum_assembly  device momentum (cut-cell + FOU advection) assembly
# amr_facegeom       device face-geometry assembly
# amr_multigrid      device multigrid V-cycle
# amr_fv_openness    device openness-folded operator
# amr_kappa_restrict / amr_kappa_dirichlet  kappa restriction / Dirichlet closure on device
# amr_pcg            MG-preconditioned CG over the device V-cycle
# amr_momentum       device momentum operator + BiCGStab vs the host AmrCutCell
# amr_flow_solver    the device AmrFlow step vs the host oracle + analytic solutions
foreach(t IN ITEMS block_octree_view amr_fv_op amr_assembly amr_momentum_assembly amr_facegeom
                   amr_multigrid amr_fv_openness amr_kappa_restrict amr_kappa_dirichlet amr_pcg
                   amr_momentum amr_flow_solver)
  _amr_test_exe(test_${t})
  peclet_amr_add_test(${t} test_${t})
endforeach()

# ---- measurement studies (label `bench`: printed tables, no correctness gate) ---------------------
# study_amr_ghost_apriori      a-priori defect study for the collocated projection (ghost port step 0)
# study_amr_seam_census        M1 seam census (docs/amr_mixed_level_cut_band_plan.md §8)
# study_amr_seam_sample_order  M2 virtual-sample order study (§8 M2)
# study_amr_seam_march         M3 seam stability probe (§8 M3; ~3 min)
foreach(t IN ITEMS study_amr_ghost_apriori study_amr_seam_census study_amr_seam_sample_order
                   study_amr_seam_march)
  _amr_test_exe(${t})
  peclet_amr_add_test(${t} ${t} LABELS bench)
endforeach()

# ---- distributed (MPI): every test at np = 1, 2, 4, 8 (np8 = a local gate) -----------------------
  # host octree / halo / adapt / rebalance / Poisson + MG, and the device-resident distributed
  # view, leaf halo, momentum, flow MG, flow step, adaptive flow and the mixed-level seam.
set(_mpi_tests amr_distributed amr_distributed_poisson amr_distributed_mg amr_distributed_fv
               amr_distributed_graded_mg amr_distributed_openness amr_distributed_adapt
               amr_distributed_rebalance amr_leaf_halo
               amr_distributed_view amr_leaf_halo_view amr_distributed_momentum
               amr_distributed_flow_mg amr_distributed_flow amr_distributed_adaptflow
               amr_distributed_seam)
foreach(t IN LISTS _mpi_tests)
  _amr_test_exe(test_${t}_mpi)
endforeach()
foreach(np IN ITEMS 1 2 4 8)
  foreach(t IN LISTS _mpi_tests)
    peclet_amr_add_mpi_test(${t}_np${np} test_${t}_mpi ${np})
  endforeach()
endforeach()
