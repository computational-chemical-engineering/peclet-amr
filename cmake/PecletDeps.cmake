# PecletDeps.cmake — self-contained dependency provisioning for peclet-amr.
#
# peclet-amr must build both ways:
#   * DEV / suite build — Kokkos comes from an installed prefix on CMAKE_PREFIX_PATH
#     (../extern/install/<backend> from tools/bootstrap_deps.sh) and the sibling headers
#     (core, morton) from ../<sibling>/include. Fast; the developer workflow.
#   * SELF-CONTAINED sdist (pip install peclet-amr) — the umbrella and siblings are ABSENT, so the
#     sibling headers are FetchContent-fetched at the suite-pinned tags and Kokkos is built
#     (OpenMP+Serial) into a staging prefix. MPI always comes from the site (mpi4py must match it).
#
# Selection is automatic (prefix/sibling present -> use it; else fetch), and can be forced off/on with
# -DPECLET_VENDOR_DEPS=ON. Keep PECLET_*_TAG in lockstep with tools/bootstrap_deps.sh and the other
# packages' PecletDeps.cmake; the release pre-flight (../tools/release/check_release_state.sh) checks them.
include_guard(GLOBAL)
include(FetchContent)

set(PECLET_KOKKOS_TAG "5.1.1" CACHE STRING "Vendored Kokkos git tag")
# peclet-amr's headers need the peclet-core SOLVER layer (peclet/core/solver/{face_csr,coloring,
# csr_operator,csr_bicgstab,vector_ops}.hpp), lifted out of the AMR tree on 2026-09-10 and first
# shipped in the core release that follows v0.6.1: until that tag exists, main builds against core's
# main (CI passes -DPECLET_CORE_TAG=main) and the pin below is repinned by the release.
set(PECLET_CORE_TAG    "v0.6.1"  CACHE STRING "Vendored core git tag (headers)")
set(PECLET_MORTON_TAG  "v0.2.1"  CACHE STRING "Vendored morton git tag (headers)")
option(PECLET_VENDOR_DEPS "Force FetchContent-build of Kokkos / fetch of the sibling headers (self-contained sdist)" OFF)

# nanobind — found via the active interpreter (scikit-build-core supplies it as a build requirement).
# MUST be a macro: find_package(Python) sets variables nanobind reads at module-creation time in the
# caller's scope.
macro(peclet_require_nanobind)
  if(NOT COMMAND nanobind_add_module)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    if(NOT nanobind_DIR)
      execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE _peclet_nb OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _peclet_nb_rc)
      if(_peclet_nb_rc EQUAL 0 AND EXISTS "${_peclet_nb}")
        set(nanobind_DIR "${_peclet_nb}")
      endif()
    endif()
    find_package(nanobind CONFIG REQUIRED)
    message(STATUS "[peclet] nanobind from ${nanobind_DIR}")
  endif()
endmacro()

# Vendored Kokkos is *installed* to a staging prefix (built once; marker file), then find_package()d —
# the same shape as tools/bootstrap_deps.sh, so the module sees one Kokkos either way.
set(PECLET_STAGE_PREFIX "${CMAKE_BINARY_DIR}/_peclet_deps" CACHE PATH "Vendored-deps staging install prefix")

function(_peclet_stage_build name url tag)  # extra -D configure args via ARGN
  FetchContent_Declare(${name} GIT_REPOSITORY "${url}" GIT_TAG "${tag}" GIT_SHALLOW TRUE)
  FetchContent_GetProperties(${name})
  if(NOT ${name}_POPULATED)
    FetchContent_Populate(${name})
  endif()
  if(EXISTS "${PECLET_STAGE_PREFIX}/.peclet_${name}_installed")
    return()
  endif()
  message(STATUS "[peclet] building+installing ${name} ${tag} -> ${PECLET_STAGE_PREFIX}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -S "${${name}_SOURCE_DIR}" -B "${${name}_BINARY_DIR}"
            -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=${PECLET_STAGE_PREFIX}"
            "-DCMAKE_PREFIX_PATH=${PECLET_STAGE_PREFIX}" -DCMAKE_CXX_STANDARD=20
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON ${ARGN}
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "[peclet] ${name} configure failed (${_rc})")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build "${${name}_BINARY_DIR}" --target install --parallel
                  RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "[peclet] ${name} build/install failed (${_rc})")
  endif()
  file(WRITE "${PECLET_STAGE_PREFIX}/.peclet_${name}_installed" "")
endfunction()

# Kokkos — prefix if present (unless forced), else vendored OpenMP+Serial installed to the staging prefix.
macro(peclet_require_kokkos)
  if(NOT PECLET_VENDOR_DEPS)
    find_package(Kokkos CONFIG QUIET)
  endif()
  if(Kokkos_FOUND)
    message(STATUS "[peclet] Kokkos ${Kokkos_VERSION} from prefix (${Kokkos_DEVICES})")
  else()
    _peclet_stage_build(kokkos "https://github.com/kokkos/kokkos.git" "${PECLET_KOKKOS_TAG}"
                        -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=ON)
    list(APPEND CMAKE_PREFIX_PATH "${PECLET_STAGE_PREFIX}")
    find_package(Kokkos CONFIG REQUIRED)
    message(STATUS "[peclet] vendored Kokkos ${Kokkos_VERSION} @ ${PECLET_STAGE_PREFIX}")
  endif()
endmacro()

# Sibling header include dir (core / morton). Returns the sibling checkout if present, else a
# FetchContent-fetched source tree's include/ (header-only — declared but not built).
function(peclet_sibling_include repo tag sibling_reldir outvar)
  set(_local "${CMAKE_CURRENT_SOURCE_DIR}/${sibling_reldir}/include")
  if(EXISTS "${_local}" AND NOT PECLET_VENDOR_DEPS)
    set(${outvar} "${_local}" PARENT_SCOPE)
    return()
  endif()
  string(TOLOWER "peclet_sib_${repo}" _name)
  FetchContent_Declare(${_name}
    GIT_REPOSITORY "https://github.com/computational-chemical-engineering/${repo}.git"
    GIT_TAG ${tag} GIT_SHALLOW TRUE)
  FetchContent_GetProperties(${_name})
  if(NOT ${_name}_POPULATED)
    FetchContent_Populate(${_name})
  endif()
  set(${outvar} "${${_name}_SOURCE_DIR}/include" PARENT_SCOPE)
  message(STATUS "[peclet] vendored ${repo} headers -> ${${_name}_SOURCE_DIR}/include")
endfunction()
