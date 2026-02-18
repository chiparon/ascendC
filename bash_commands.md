git config --global http.version HTTP/1.1
git clone https://github.com/chiparon/ascendC.git

cd ascendC/13_matmulleakyrelu_kernellaunch/Optimized/v2_algorithm/MatmulLeakyReluInvocationAsync
bash run.sh -r npu \
  -d /tmp/matmul_opt_npu_build \
  -p /tmp/matmul_opt_npu_out \
  --build-only


#s1234
bash run.sh -r npu -v Ascend910B3 -d /tmp/matmul_opt_npu_build -p /tmp/matmul_opt_npu_out \
  --m 2048 --n 2048 --k 2048 --repeat 10 --force-core 2 --run-only
  
bash run.sh -r npu -v Ascend910B3 -d /tmp/matmul_opt_npu_build -p /tmp/matmul_opt_npu_out \
  --m 2048 --n 2048 --k 2048 --repeat 10 --force-core 0 --run-only
  
bash run.sh -r npu -v Ascend910B3 -d /tmp/matmul_opt_npu_build -p /tmp/matmul_opt_npu_out \
  --m 4096 --n 1024 --k 4096 --repeat 10 --force-core 2 --run-only
  
bash run.sh -r npu -v Ascend910B3 -d /tmp/matmul_opt_npu_build -p /tmp/matmul_opt_npu_out \
  --m 4096 --n 1024 --k 4096 --repeat 10 --force-core 0 --run-only
  
bash run.sh -r npu -v Ascend910B3 -d /tmp/matmul_opt_npu_build -p /tmp/matmul_opt_npu_out \
  --m 1024 --n 512 --k 1024 --repeat 10 --force-core 2 --run-only
  
bash run.sh -r npu -v Ascend910B3 -d /tmp/matmul_opt_npu_build -p /tmp/matmul_opt_npu_out \
  --m 1024 --n 512 --k 1024 --repeat 10 --force-core 0 --run-only
  
bash run.sh -r npu -v Ascend910B3 -d /tmp/matmul_opt_npu_build -p /tmp/matmul_opt_npu_out \
  --m 512 --n 128 --k 512 --repeat 10 --force-core 0 --run-only
echo ASCEND_HOME_PATH

