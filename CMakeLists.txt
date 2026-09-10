cmake_minimum_required(VERSION 3.24)
# --- Version: single-sourced from pyproject.toml (suite/docs/QUALITY_PLAN.md D4) ---
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _peclet_amr_version_line REGEX "^version = \"")
string(REGEX MATCH "[0-9]+(\\.[0-9]+)*" PECLET_AMR_VERSION "${_peclet_amr_version_line}")
if(NOT PECLET_AMR_VERSION)
  message(FATAL_ERROR "peclet-amr: could not read `version = \"x.y.z\"` from pyproject.toml")
endif()
project(peclet_amr VERSION ${PECLET_AMR_VERSION} LANGUAGES CXX)

# peclet-amr — the block-local-Morton AMR octree and the collocated-projection Navier–Stokes solver
# on it (peclet::amr; Python `peclet.amr`), relocated out of peclet-core with its history on
# 2026-09-10 (suite/docs/QUALITY_PLAN.md D6 / G.2). Header-only C++20 over peclet-core (the
# decomposition, halos, geometry and the face-CSR solver layer) and peclet-morton (Z-order codes),
# compiled through Kokkos (CUDA / HIP / OpenMP — the backend is whatever prefix CMAKE_PREFIX_PATH
# points at, e.g. ../extern/install/host-openmp from ../tools/bootstrap_deps.sh) over MPI.
#
# Build (the suite layout, core + morton as siblings):
#   cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp" \
#         -DPECLET_AMR_BUILD_TESTS=ON      # MPI + a Kokkos prefix are required
#   cmake --build build -j       ->  build/peclet/amr/_amr.*.so  (import peclet.amr)
#   ctest --test-dir build --output-on-failure -LE bench

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# MPI is REQUIRED: the distributed octree / flow (LeafHalo, DistributedOctree, rebalance, the
# multi-rank step) and the Python module (whose DistributedOctree runs over MPI_COMM_WORLD) link
# it, and the device flow reaches core's Kokkos grid halo, which has no no-MPI stub (core's
# mpi_stub.hpp covers the host halo only — a Kokkos + no-MPI build was never a core configuration).
# The Python module `peclet.amr` (nanobind, found via the active interpreter).
option(PECLET_AMR_BUILD_PYTHON "Build the peclet.amr Python module" ON)
# Tests: registers the C++ battery (tests/), the benchmarks + measurement studies (label `bench`)
# and the Python ctests (test_amr.py serial + np=2, state_hash.py). OFF by default so a
# `pip install .` is unaffected; ON in CI and the dev recipe.
option(PECLET_AMR_BUILD_TESTS "Build and register the peclet-amr test battery (ctest)" OFF)
# Redistributable single-GPU CUDA wheel: libcudart from the `nvidia-cuda-runtime` PyPI wheel.
option(PECLET_CUDA_RUNTIME_WHEEL "RPATH the module to the nvidia-cuda-runtime wheel's libcudart" OFF)

# --- Dependencies (cmake/PecletDeps.cmake): Kokkos from the prefix (or vendored), the core + morton
#     headers from the sibling checkouts ../core and ../morton (or fetched at their pins). ---
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PecletDeps)
peclet_require_kokkos()
peclet_sibling_include(peclet-core   "${PECLET_CORE_TAG}"   "../core"   PECLET_CORE_INCLUDE)
peclet_sibling_include(peclet-morton "${PECLET_MORTON_TAG}" "../morton" PECLET_MORTON_INCLUDE)

# --- Header-only library target: peclet::amr = include/ + core + morton + Kokkos (+ MPI). ---
add_library(peclet_amr INTERFACE)
add_library(peclet::amr ALIAS peclet_amr)
target_include_directories(peclet_amr INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PECLET_CORE_INCLUDE}>
  $<BUILD_INTERFACE:${PECLET_MORTON_INCLUDE}>
  $<INSTALL_INTERFACE:include>)
target_compile_features(peclet_amr INTERFACE cxx_std_20)
# morton in Kokkos mode: MORTON_HD resolves to KOKKOS_FUNCTION, so the block octree, the cut-cell
# closures and the face-CSR row kernels are device-callable on every backend.
target_link_libraries(peclet_amr INTERFACE Kokkos::kokkos)
target_compile_definitions(peclet_amr INTERFACE MORTON_ENABLE_KOKKOS=1)

find_package(MPI REQUIRED COMPONENTS CXX)
# Pin MPIEXEC_EXECUTABLE to the launcher next to mpicxx (cmake/PecletAmrPinMpiexec.cmake): a
# foreign launcher on PATH silently runs every rank as a singleton.
include(PecletAmrPinMpiexec)
target_link_libraries(peclet_amr INTERFACE MPI::MPI_CXX)
message(STATUS "peclet-amr: MPI ${MPI_CXX_COMPILER}")

# --- Python module: peclet.amr (private extension `_amr`, re-exported by peclet/amr/__init__.py,
#     kept as a plain file under packaging/ and staged into <build>/peclet/amr so
#     `PYTHONPATH=<build> python …` works in the dev loop). ---
if(PECLET_AMR_BUILD_PYTHON)
  peclet_require_nanobind()
  # NB_STATIC: bundle nanobind's runtime. NOMINSIZE: nanobind's default -Os is rejected by nvcc
  # ("'s': expected a number") since the Kokkos device sources compile as CXX through the launch
  # compiler.
  nanobind_add_module(amr NB_STATIC NOMINSIZE python/amr_bindings.cpp)
  set_target_properties(amr PROPERTIES OUTPUT_NAME _amr   # -> peclet.amr._amr (NB_MODULE(_amr))
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/peclet/amr")
  target_link_libraries(amr PRIVATE peclet_amr)
  # `peclet.amr.build_toolchain`: the byte gate (python/state_hash.py) compares hashes only between
  # builds of one toolchain — the last bits of a Krylov solve depend on the compiler, its version and
  # the optimisation level — and SKIPS (77) against a reference recorded elsewhere.
  target_compile_definitions(amr PRIVATE
    "PECLET_AMR_BUILD_TOOLCHAIN=\"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} ${CMAKE_BUILD_TYPE} ${CMAKE_SYSTEM_PROCESSOR} Kokkos ${Kokkos_VERSION}\"")
  # -mfma on the HOST halves of the MODULE only (docs/amr_setup_parallel_plan.md rung 0b): the
  # scene layer's fma-canonical expressions otherwise compile std::fma to a libm CALL — measured
  # ~15% of setSolid at one thread. nvcc route: -Xcompiler. std::fma itself is correctly rounded
  # either way; the flag also lets gcc CONTRACT ordinary a*b+c, which is why it stays off the test
  # binaries (amr_multigrid / amr_fv_openness gate host vs device BIT-exact on host backends — with
  # the flag on both halves they drift by 1 ulp) — exactly the placement peclet-core had.
  target_compile_options(amr PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-mfma>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mfma>)
  if(Kokkos_ENABLE_HIP)
    # HIP/lld: nanobind's hidden visibility + --gc-sections drop Kokkos vtables the host objects
    # reference (see flow/CMakeLists.txt); default visibility, keep sections, on the HIP path only.
    set_target_properties(amr PROPERTIES CXX_VISIBILITY_PRESET default)
    target_link_options(amr PRIVATE -Wl,--no-gc-sections)
  endif()
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/packaging/amr_init.py
                 ${CMAKE_CURRENT_BINARY_DIR}/peclet/amr/__init__.py COPYONLY)
  # Type stub (packaging/_amr.pyi, generated by `python -m nanobind.stubgen -m peclet.amr._amr`
  # against the built module — regenerate after changing a binding), staged beside the extension
  # with the PEP 561 marker.
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/packaging/_amr.pyi
                 ${CMAKE_CURRENT_BINARY_DIR}/peclet/amr/_amr.pyi COPYONLY)
  file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/peclet/amr/py.typed)
  if(PECLET_CUDA_RUNTIME_WHEEL)
    set_target_properties(amr PROPERTIES
      INSTALL_RPATH "$ORIGIN/../../nvidia/cu13/lib" INSTALL_RPATH_USE_LINK_PATH OFF)
  endif()
  if(DEFINED SKBUILD)
    install(TARGETS amr LIBRARY DESTINATION peclet/amr COMPONENT python)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/amr_init.py
            DESTINATION peclet/amr RENAME __init__.py COMPONENT python)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/_amr.pyi
                  ${CMAKE_CURRENT_BINARY_DIR}/peclet/amr/py.typed
            DESTINATION peclet/amr COMPONENT python)
  endif()
  message(STATUS "peclet-amr: Python module peclet.amr (Kokkos ${Kokkos_VERSION}, ${Kokkos_DEVICES})")
endif()

# --- Install: the headers + an exported header-only target (find_package(peclet-amr CONFIG)). The
#     core / morton includes are BUILD_INTERFACE-only; an installed consumer supplies its own. ---
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set_target_properties(peclet_amr PROPERTIES EXPORT_NAME amr)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS peclet_amr EXPORT peclet-amrTargets)
install(EXPORT peclet-amrTargets NAMESPACE peclet::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/peclet-amr)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/peclet-amrConfigVersion.cmake"
                                 COMPATIBILITY SameMajorVersion)
configure_package_config_file(cmake/peclet-amrConfig.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/peclet-amrConfig.cmake"
                              INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/peclet-amr)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/peclet-amrConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/peclet-amrConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/peclet-amr)

# --- Tests + benchmarks: ONE registration protocol (cmake/PecletAmrTest.cmake: SKIP_RETURN_CODE 77,
#     labels mpi / np8 / bench / python). ---
if(PECLET_AMR_BUILD_TESTS)
  enable_testing()
  include(PecletAmrTest)
  add_subdirectory(tests)
  add_subdirectory(benchmarks)
  if(TARGET amr)
    # The Python ctests run against the module staged in this build tree (PYTHONPATH=<build>).
    # PECLET_AMR_TEST_NP tells the script how many ranks ctest launched; it exits non-zero if
    # comm.size differs, so a launcher/mpi4py mismatch (N singletons) is a FAILURE, not a pass.
    set(_py_env "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
    add_test(NAME python_amr COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_amr.py")
    set_tests_properties(python_amr PROPERTIES LABELS "python"
      ENVIRONMENT "${_py_env};PECLET_AMR_TEST_NP=1")
    add_test(NAME python_amr_np2
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                     "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_amr.py" ${MPIEXEC_POSTFLAGS})
    set_tests_properties(python_amr_np2 PROPERTIES LABELS "mpi;python"
      ENVIRONMENT "${_py_env};PECLET_AMR_TEST_NP=2")
    # The structural byte gate (QUALITY_PLAN §3.G): fixed-seed reference runs of every public entry
    # path, hashed; `--check python/state_hash_reference.json` fails on any drift. The reference is
    # toolchain-specific (its `toolchain` key); a build of another compiler / version / build type
    # exits 77 and ctest reports the gate SKIPPED (CI), never passed.
    add_test(NAME python_state_hash
             COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/state_hash.py"
                     --check "${CMAKE_CURRENT_SOURCE_DIR}/python/state_hash_reference.json")
    set_tests_properties(python_state_hash PROPERTIES LABELS "python" SKIP_RETURN_CODE 77
      ENVIRONMENT "${_py_env};OMP_NUM_THREADS=1")
  endif()
endif()
