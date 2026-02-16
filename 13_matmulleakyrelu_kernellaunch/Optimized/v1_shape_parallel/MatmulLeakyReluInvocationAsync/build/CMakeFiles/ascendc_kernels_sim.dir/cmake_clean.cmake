file(REMOVE_RECURSE
  "lib/libascendc_kernels_sim.pdb"
  "lib/libascendc_kernels_sim.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/ascendc_kernels_sim.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
