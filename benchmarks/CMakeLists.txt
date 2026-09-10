# Benchmarks are registered as ctests with the label `bench` so `ctest -N` lists them and
# `ctest -L bench` runs them at a small size; they are excluded from the correctness battery
# (`ctest -LE bench`) because they print timings and gate nothing.

# AMR flow / Poisson host-vs-device benchmark (Kokkos + morton).
add_executable(bench_amr_flow bench_amr_flow.cpp)
target_link_libraries(bench_amr_flow PRIVATE peclet::amr)
peclet_amr_add_test(bench_amr_flow bench_amr_flow LABELS bench ARGS 4 2 3)   # Lmax_device, steps, Lmax_host
