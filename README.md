![dabun logo](/assets/logo/icononly_transparent_nobuffer.png)

## Cleanup of include/

- [ ] include/dabun/aligned_wrapper.hpp
- [ ] include/dabun/arm/arithmetic_operation.hpp
- [ ] include/dabun/arm/configuration.hpp
- [ ] include/dabun/arm/elementwise_operation.hpp
- [ ] include/dabun/arm/loop_nest.hpp
- [ ] include/dabun/arm/loop_nest_fp16.hpp
- [ ] include/dabun/arm/meta_mnemonics.hpp
- [x] include/dabun/arm/multi_vreg.hpp
- [x] include/dabun/arm/peak_gflops.hpp
- [ ] include/dabun/arm/transposer.hpp
- [ ] include/dabun/arm/xbyak.hpp
- [ ] include/dabun/check.hpp
- [ ] include/dabun/common.hpp
- [ ] include/dabun/core.hpp
- [ ] include/dabun/hask/number_array.hpp
- [ ] include/dabun/hask/random.hpp
- [ ] include/dabun/hask/type_traits.hpp
- [ ] include/dabun/isa.hpp
- [ ] include/dabun/loop_nest.hpp
- [ ] include/dabun/loop_tree/all_nodes.hpp
- [ ] include/dabun/loop_tree/compiled_loop_nest_node.hpp
- [ ] include/dabun/loop_tree/compiled_transpose_node.hpp
- [ ] include/dabun/loop_tree/compute_node.hpp
- [ ] include/dabun/loop_tree/for_loop_node.hpp
- [ ] include/dabun/loop_tree/nested_for_loops_node.hpp
- [ ] include/dabun/loop_tree/node.hpp
- [ ] include/dabun/loop_tree/program.hpp
- [ ] include/dabun/loop_tree/report.hpp
- [ ] include/dabun/loop_tree/transpose_node.hpp
- [ ] include/dabun/loop_tree/types.hpp
- [ ] include/dabun/loop_tree/utility.hpp
- [ ] include/dabun/one_constant.hpp
- [x] include/dabun/peak_gflops.hpp
- [x] include/dabun/peak_gflops.ipp
- [ ] include/dabun/predef.hpp
- [ ] include/dabun/random_vector.hpp
- [ ] include/dabun/serialization.hpp
- [ ] include/dabun/utility/array.hpp
- [ ] include/dabun/utility/for_all.hpp
- [ ] include/dabun/utility/vek.hpp
- [ ] include/dabun/x86/address_packer.hpp
- [ ] include/dabun/x86/aot_perf.hpp
- [ ] include/dabun/x86/arithmetic_operation.hpp
- [ ] include/dabun/x86/configuration.hpp
- [ ] include/dabun/x86/denormals.hpp
- [ ] include/dabun/x86/elementwise_operation.hpp
- [ ] include/dabun/x86/loop_nest.hpp
- [x] include/dabun/x86/multi_vmm.hpp
- [ ] include/dabun/x86/oprof-jitdump.hpp
- [x] include/dabun/x86/peak_gflops.hpp
- [ ] include/dabun/x86/transposer.hpp
- [ ] include/dabun/x86/xbyak.hpp
- [x] include/dabun/aligned_vector.hpp
- [x] include/dabun/amx/amx_loop_nest.hpp (removed)
- [x] include/dabun/arithmetic_operation.hpp
- [x] include/dabun/bf16x2.hpp (removed)
- [x] include/dabun/code_generator.hpp
- [x] include/dabun/code_generator/aot_fn.hpp
- [x] include/dabun/code_generator/code_generator.hpp
- [x] include/dabun/code_generator/memory_resource.hpp
- [x] include/dabun/code_generator/xbyak.hpp
- [x] include/dabun/configuration.hpp
- [x] include/dabun/elementwise_operation.hpp
- [x] include/dabun/fmt.hpp (removed)
- [x] include/dabun/hask/aligned_alloc.hpp (moved to sysml)
- [x] include/dabun/hask/apple.hpp
- [x] include/dabun/hask/miltuple.hpp (moved to sysml)
- [x] include/dabun/intcmp.hpp (moved to sysml)
- [x] include/dabun/loop_nest_descriptor.hpp
- [x] include/dabun/math.hpp
- [x] include/dabun/measure.hpp (moved to sysml)
- [x] include/dabun/mpl.hpp (removed)
- [x] include/dabun/mpl/cond.hpp (removed)
- [x] include/dabun/mpl/core.hpp (removed)
- [x] include/dabun/namespace.hpp (moved content to isa.hpp)
- [x] include/dabun/numeric.hpp
- [x] include/dabun/peak_gflops.ipp
- [x] include/dabun/qvec4.hpp (removed)
- [x] include/dabun/third_party/biovault_bfloat16.hpp (removed)
- [x] include/dabun/third_party/half.hpp (removed)
- [x] include/dabun/thread/barrier.hpp
- [x] include/dabun/thread/core.hpp
- [x] include/dabun/thread/cpu_pool.hpp
- [x] include/dabun/thread/cpu_set.hpp
- [x] include/dabun/thread/operating_cpu_set.hpp
- [x] include/dabun/thread/parallel_for.hpp
- [x] include/dabun/thread/semi_dynamic_task_queue.hpp (removed)
- [x] include/dabun/transposer.hpp
- [x] include/dabun/utility/log.hpp
- [x] include/dabun/utility/most_frequent_queue.hpp
- [x] include/dabun/utility/tmp_file_name.hpp

## Cleanup of src/

- [ ] src/loop_nest.cpp
- [ ] src/transposer.cpp
- [ ] src/x86/multi_vmm.cpp
- [ ] src/peak_gflops.cpp