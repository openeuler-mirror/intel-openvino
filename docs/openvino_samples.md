# 1. Prerequisite

OpenVINO has been natively integrated into openEuler starting from openEuler 24.03 LTS SP1.

# 2. Install Intel GPU drivers and compute runtime libraries

The OpenVINO dependencies, Intel GPUs drivers and graphics compute runtime have been integrated into openEuler 24.03 LTS SP1, please install the required packages from openEuler repository:

```shell
# sudo dnf install -y intel-gmmlib intel-gsc intel-igc-cm intel-igc-core intel-igc-opencl \
intel-level-zero-gpu intel-ocloc intel-opencl level-zero libmetee ocl-icd
```

# 3. Install OpenVINO packages from openEuler repository

- List all OpenVINO packages

```shell
# sudo dnf list *openvino*
```

The available openvino packages:
```
libopenvino.x86_64                                   2024.3.0-1.oe2403           @oe-openvino
libopenvino-auto-batch-plugin.x86_64                 2024.3.0-1.oe2403           @oe-openvino
libopenvino-auto-plugin.x86_64                       2024.3.0-1.oe2403           @oe-openvino
libopenvino-devel.x86_64                             2024.3.0-1.oe2403           @oe-openvino
libopenvino-intel-cpu-plugin.x86_64                  2024.3.0-1.oe2403           @oe-openvino
libopenvino-intel-gpu-plugin.x86_64                  2024.3.0-1.oe2403           @oe-openvino
libopenvino_ir_frontend.x86_64                       2024.3.0-1.oe2403           @oe-openvino
libopenvino_onnx_frontend.x86_64                     2024.3.0-1.oe2403           @oe-openvino
libopenvino_paddle_frontend.x86_64                   2024.3.0-1.oe2403           @oe-openvino
libopenvino_pytorch_frontend.x86_64                  2024.3.0-1.oe2403           @oe-openvino
libopenvino_tensorflow_frontend.x86_64               2024.3.0-1.oe2403           @oe-openvino
libopenvino_tensorflow_lite_frontend.x86_64          2024.3.0-1.oe2403           @oe-openvino
openvino-samples.noarch                              2024.3.0-1.oe2403           @oe-openvino
libopenvino-hetero-plugin.x86_64                     2024.3.0-1.oe2403           oe-openvino
openvino.src                                         2024.3.0-1.oe2403           oe-openvino
```
- Install basic OpenVINO packages

```shell
# sudo dnf install -y libopenvino libopenvino-intel-cpu-plugin libopenvino-intel-gpu-plugin openvino-samples libopenvino-devel
```

# 4. Build OpenVINO samples
- Install required build tools and dependencies

```shell
# sudo dnf install -y cmake gcc g++ wget
# sudo dnf install -y opencl-headers opencl-clhpp ocl-icd-devel
```
- Build the sample code

```shell
# cd /usr/share/openvino/samples/cpp/
# ./build_samples.sh
```

The build results:

```
Setting environment variables for building samples...
-- The C compiler identification is GNU 12.3.1
-- The CXX compiler identification is GNU 12.3.1
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
CMake Deprecation Warning at thirdparty/gflags/gflags/CMakeLists.txt:73 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


-- Looking for C++ include unistd.h
-- Looking for C++ include unistd.h - found
-- Looking for C++ include stdint.h
-- Looking for C++ include stdint.h - found
-- Looking for C++ include sys/types.h
-- Looking for C++ include sys/types.h - found
-- Looking for C++ include fnmatch.h
-- Looking for C++ include fnmatch.h - found
-- Looking for strtoll
-- Looking for strtoll - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- Using the multi-header code from /usr/share/openvino/samples/cpp/thirdparty/nlohmann_json/include/
-- Configuring done (0.7s)
-- Generating done (0.0s)
-- Build files have been written to: /home/juntian/openvino_cpp_samples_build
[  2%] Linking CXX static library ../../../intel64/Release/libgflags_nothreads.a
[ 10%] Built target gflags_nothreads_static
[ 13%] Linking CXX static library ../../intel64/Release/libie_samples_utils.a
[ 21%] Built target ie_samples_utils
[ 29%] Linking CXX executable ../intel64/Release/hello_query_device
[ 29%] Linking CXX executable ../../intel64/Release/throughput_benchmark
[ 29%] Linking CXX executable ../../intel64/Release/sync_benchmark
[ 32%] Linking CXX static library ../../intel64/Release/libformat_reader.a
[ 48%] Built target format_reader
[ 59%] Linking CXX executable ../intel64/Release/model_creation_sample
[ 59%] Linking CXX executable ../intel64/Release/hello_classification
[ 59%] Linking CXX executable ../intel64/Release/hello_nv12_input_classification
[ 59%] Linking CXX executable ../intel64/Release/classification_sample_async
[ 62%] Linking CXX executable ../intel64/Release/benchmark_app
[ 70%] Built target hello_query_device
[ 70%] Built target throughput_benchmark
[ 70%] Built target sync_benchmark
[ 72%] Linking CXX executable ../intel64/Release/hello_reshape_ssd
[ 75%] Built target model_creation_sample
[ 83%] Built target hello_nv12_input_classification
[ 83%] Built target hello_classification
[ 83%] Built target classification_sample_async
[ 97%] Built target benchmark_app
[100%] Built target hello_reshape_ssd
[100%] Built target ov_samples

Build completed, you can find binaries for all samples in the /home/emotional_openeuler_robot/openvino_cpp_samples_build/intel64/Release subfolder.
```

# 5. Run OpenVINO samples and benchmark tools
- Run hello_query_device for listing the supported devices
```shell
# cd ~/openvino_cpp_samples_build/intel64/Release
# ./hello_query_device
```

It will list the avaliable OpenVINO devices like CPU, GPU, NPU, etc.

```
[ INFO ] Build ................................. 2024.3.0-1-1e3b88e4e3f
[ INFO ]
[ INFO ] Available devices:
[ INFO ] CPU
[ INFO ] 	SUPPORTED_PROPERTIES:
[ INFO ] 		Immutable: AVAILABLE_DEVICES : ""
[ INFO ] 		Immutable: RANGE_FOR_ASYNC_INFER_REQUESTS : 1 1 1
[ INFO ] 		Immutable: RANGE_FOR_STREAMS : 1 16
[ INFO ] 		Immutable: EXECUTION_DEVICES : CPU
[ INFO ] 		Immutable: FULL_DEVICE_NAME : 11th Gen Intel(R) Core(TM) i9-11900K @ 3.50GHz
[ INFO ] 		Immutable: OPTIMIZATION_CAPABILITIES : WINOGRAD FP32 INT8 BIN EXPORT_IMPORT
[ INFO ] 		Immutable: DEVICE_TYPE : integrated
[ INFO ] 		Immutable: DEVICE_ARCHITECTURE : intel64
[ INFO ] 		Mutable: NUM_STREAMS : 1
[ INFO ] 		Mutable: INFERENCE_NUM_THREADS : 0
[ INFO ] 		Mutable: PERF_COUNT : NO
[ INFO ] 		Mutable: INFERENCE_PRECISION_HINT : f32
[ INFO ] 		Mutable: PERFORMANCE_HINT : LATENCY
[ INFO ] 		Mutable: EXECUTION_MODE_HINT : PERFORMANCE
[ INFO ] 		Mutable: PERFORMANCE_HINT_NUM_REQUESTS : 0
[ INFO ] 		Mutable: ENABLE_CPU_PINNING : YES
[ INFO ] 		Mutable: SCHEDULING_CORE_TYPE : ANY_CORE
[ INFO ] 		Mutable: MODEL_DISTRIBUTION_POLICY : ""
[ INFO ] 		Mutable: ENABLE_HYPER_THREADING : YES
[ INFO ] 		Mutable: DEVICE_ID : ""
[ INFO ] 		Mutable: CPU_DENORMALS_OPTIMIZATION : NO
[ INFO ] 		Mutable: LOG_LEVEL : LOG_NONE
[ INFO ] 		Mutable: CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE : 1
[ INFO ] 		Mutable: DYNAMIC_QUANTIZATION_GROUP_SIZE : 32
[ INFO ] 		Mutable: KV_CACHE_PRECISION : f16
[ INFO ] 		Mutable: AFFINITY : CORE
[ INFO ]
[ INFO ] GPU.0
[ INFO ] 	SUPPORTED_PROPERTIES:
[ INFO ] 		Immutable: AVAILABLE_DEVICES : 0 1
[ INFO ] 		Immutable: RANGE_FOR_ASYNC_INFER_REQUESTS : 1 2 1
[ INFO ] 		Immutable: RANGE_FOR_STREAMS : 1 2
[ INFO ] 		Immutable: OPTIMAL_BATCH_SIZE : 1
[ INFO ] 		Immutable: MAX_BATCH_SIZE : 1
[ INFO ] 		Immutable: DEVICE_ARCHITECTURE : GPU: vendor=0x8086 arch=v12.1.0
[ INFO ] 		Immutable: FULL_DEVICE_NAME : Intel(R) UHD Graphics 750 (iGPU)
[ INFO ] 		Immutable: DEVICE_UUID : 86808a4c040000000002000000000000
[ INFO ] 		Immutable: DEVICE_LUID : 409a0000499a0000
[ INFO ] 		Immutable: DEVICE_TYPE : integrated
[ INFO ] 		Immutable: DEVICE_GOPS : {f16:1331.2,f32:665.6,i8:2662.4,u8:2662.4}
[ INFO ] 		Immutable: OPTIMIZATION_CAPABILITIES : FP32 BIN FP16 INT8 EXPORT_IMPORT
[ INFO ] 		Immutable: GPU_DEVICE_TOTAL_MEM_SIZE : 26265014272
[ INFO ] 		Immutable: GPU_UARCH_VERSION : 12.1.0
[ INFO ] 		Immutable: GPU_EXECUTION_UNITS_COUNT : 32
[ INFO ] 		Immutable: GPU_MEMORY_STATISTICS : ""
[ INFO ] 		Mutable: PERF_COUNT : NO
[ INFO ] 		Mutable: MODEL_PRIORITY : MEDIUM
[ INFO ] 		Mutable: GPU_HOST_TASK_PRIORITY : MEDIUM
[ INFO ] 		Mutable: GPU_QUEUE_PRIORITY : MEDIUM
[ INFO ] 		Mutable: GPU_QUEUE_THROTTLE : MEDIUM
[ INFO ] 		Mutable: GPU_ENABLE_SDPA_OPTIMIZATION : YES
[ INFO ] 		Mutable: GPU_ENABLE_LOOP_UNROLLING : YES
[ INFO ] 		Mutable: GPU_DISABLE_WINOGRAD_CONVOLUTION : NO
[ INFO ] 		Mutable: CACHE_DIR : ""
[ INFO ] 		Mutable: CACHE_MODE : optimize_speed
[ INFO ] 		Mutable: PERFORMANCE_HINT : LATENCY
[ INFO ] 		Mutable: EXECUTION_MODE_HINT : PERFORMANCE
[ INFO ] 		Mutable: COMPILATION_NUM_THREADS : 16
[ INFO ] 		Mutable: NUM_STREAMS : 1
[ INFO ] 		Mutable: PERFORMANCE_HINT_NUM_REQUESTS : 0
[ INFO ] 		Mutable: INFERENCE_PRECISION_HINT : f16
[ INFO ] 		Mutable: ENABLE_CPU_PINNING : NO
[ INFO ] 		Mutable: DEVICE_ID : 0
[ INFO ] 		Mutable: DYNAMIC_QUANTIZATION_GROUP_SIZE : 0
[ INFO ]
[ INFO ] GPU.1
[ INFO ] 	SUPPORTED_PROPERTIES:
[ INFO ] 		Immutable: AVAILABLE_DEVICES : 0 1
[ INFO ] 		Immutable: RANGE_FOR_ASYNC_INFER_REQUESTS : 1 2 1
[ INFO ] 		Immutable: RANGE_FOR_STREAMS : 1 2
[ INFO ] 		Immutable: OPTIMAL_BATCH_SIZE : 1
[ INFO ] 		Immutable: MAX_BATCH_SIZE : 1
[ INFO ] 		Immutable: DEVICE_ARCHITECTURE : GPU: vendor=0x8086 arch=v12.55.8
[ INFO ] 		Immutable: FULL_DEVICE_NAME : Intel(R) Arc(TM) A770 Graphics (dGPU)
[ INFO ] 		Immutable: DEVICE_UUID : 8680a056080000000300000000000000
[ INFO ] 		Immutable: DEVICE_LUID : 38474cccfc7f0000
[ INFO ] 		Immutable: DEVICE_TYPE : discrete
[ INFO ] 		Immutable: DEVICE_GOPS : {f16:0,f32:19660.8,i8:0,u8:0}
[ INFO ] 		Immutable: OPTIMIZATION_CAPABILITIES : FP32 BIN FP16 INT8 GPU_HW_MATMUL EXPORT_IMPORT
[ INFO ] 		Immutable: GPU_DEVICE_TOTAL_MEM_SIZE : 16225243136
[ INFO ] 		Immutable: GPU_UARCH_VERSION : 12.55.8
[ INFO ] 		Immutable: GPU_EXECUTION_UNITS_COUNT : 512
[ INFO ] 		Immutable: GPU_MEMORY_STATISTICS : ""
[ INFO ] 		Mutable: PERF_COUNT : NO
[ INFO ] 		Mutable: MODEL_PRIORITY : MEDIUM
[ INFO ] 		Mutable: GPU_HOST_TASK_PRIORITY : MEDIUM
[ INFO ] 		Mutable: GPU_QUEUE_PRIORITY : MEDIUM
[ INFO ] 		Mutable: GPU_QUEUE_THROTTLE : MEDIUM
[ INFO ] 		Mutable: GPU_ENABLE_SDPA_OPTIMIZATION : YES
[ INFO ] 		Mutable: GPU_ENABLE_LOOP_UNROLLING : YES
[ INFO ] 		Mutable: GPU_DISABLE_WINOGRAD_CONVOLUTION : NO
[ INFO ] 		Mutable: CACHE_DIR : ""
[ INFO ] 		Mutable: CACHE_MODE : optimize_speed
[ INFO ] 		Mutable: PERFORMANCE_HINT : LATENCY
[ INFO ] 		Mutable: EXECUTION_MODE_HINT : PERFORMANCE
[ INFO ] 		Mutable: COMPILATION_NUM_THREADS : 16
[ INFO ] 		Mutable: NUM_STREAMS : 1
[ INFO ] 		Mutable: PERFORMANCE_HINT_NUM_REQUESTS : 0
[ INFO ] 		Mutable: INFERENCE_PRECISION_HINT : f16
[ INFO ] 		Mutable: ENABLE_CPU_PINNING : NO
[ INFO ] 		Mutable: DEVICE_ID : 1
[ INFO ] 		Mutable: DYNAMIC_QUANTIZATION_GROUP_SIZE : 0
[ INFO ]
```

- Run benchmark tools to evaluate the latency and throughput

Download machine learning models from Intel Open Model Zoo

```shell
# wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/asl-recognition-0004/FP16/asl-recognition-0004.xml
# wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/asl-recognition-0004/FP16/asl-recognition-0004.bin
```

Run benchmark on CPU to evaluate latency

```shell
# ./benchmark_app -m asl-recognition-0004.xml -d CPU -hint latency
```

The benchmark result on CPU for the latency

```
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
[Step 2/11] Loading OpenVINO Runtime
[ INFO ] OpenVINO:
[ INFO ] Build ................................. 2024.3.0-1-1e3b88e4e3f
[ INFO ]
[ INFO ] Device info:
[ INFO ] CPU
[ INFO ] Build ................................. 2024.3.0-1-1e3b88e4e3f
[ INFO ]
[ INFO ]
[Step 3/11] Setting device configuration
[Step 4/11] Reading model files
[ INFO ] Loading model files
[ INFO ] Read model took 8.93 ms
[ INFO ] Original model I/O parameters:
[ INFO ] Network inputs:
[ INFO ]     input (node: input) : f32 / [N,C,D,H,W] / [1,3,16,224,224]
[ INFO ] Network outputs:
[ INFO ]     output (node: output) : f32 / [...] / [1,100]
[Step 5/11] Resizing model to match image sizes and given batch
[Step 6/11] Configuring input of the model
[ INFO ] Model batch size: 1
[ INFO ] Network inputs:
[ INFO ]     input (node: input) : f32 / [N,C,D,H,W] / [1,3,16,224,224]
[ INFO ] Network outputs:
[ INFO ]     output (node: output) : f32 / [...] / [1,100]
[Step 7/11] Loading the model to the device
[ INFO ] Compile model took 91.66 ms
[Step 8/11] Querying optimal runtime parameters
[ INFO ] Model:
[ INFO ]   NETWORK_NAME: torch-jit-export
[ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
[ INFO ]   NUM_STREAMS: 1
[ INFO ]   INFERENCE_NUM_THREADS: 8
[ INFO ]   PERF_COUNT: NO
[ INFO ]   INFERENCE_PRECISION_HINT: f32
[ INFO ]   PERFORMANCE_HINT: LATENCY
[ INFO ]   EXECUTION_MODE_HINT: PERFORMANCE
[ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
[ INFO ]   ENABLE_CPU_PINNING: YES
[ INFO ]   SCHEDULING_CORE_TYPE: ANY_CORE
[ INFO ]   MODEL_DISTRIBUTION_POLICY:
[ INFO ]   ENABLE_HYPER_THREADING: NO
[ INFO ]   EXECUTION_DEVICES: CPU
[ INFO ]   CPU_DENORMALS_OPTIMIZATION: NO
[ INFO ]   LOG_LEVEL: LOG_NONE
[ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1
[ INFO ]   DYNAMIC_QUANTIZATION_GROUP_SIZE: 32
[ INFO ]   KV_CACHE_PRECISION: f16
[ INFO ]   AFFINITY: CORE
[Step 9/11] Creating infer requests and preparing input tensors
[ WARNING ] No input files were given: all inputs will be filled with random values!
[ INFO ] Test Config 0
[ INFO ] input  ([N,C,D,H,W], f32, [1,3,16,224,224], static):	random (binary data/numpy array is expected)
[Step 10/11] Measuring performance (Start inference asynchronously, 1 inference requests, limits: 60000 ms duration)
[ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
[ INFO ] First inference took 35.35 ms
[Step 11/11] Dumping statistics report
[ INFO ] Execution Devices: [ CPU ]
[ INFO ] Count:               2273 iterations
[ INFO ] Duration:            60034.21 ms
[ INFO ] Latency:
[ INFO ]    Median:           26.26 ms
[ INFO ]    Average:          26.40 ms
[ INFO ]    Min:              25.29 ms
[ INFO ]    Max:              35.82 ms
[ INFO ] Throughput:          37.86 FPS
```

Run benchmark on GPU to evaluate throughput

```shell
# ./benchmark_app -m asl-recognition-0004.xml -d GPU.1 -hint throughput
```

The benchmark result on GPU for the throughput

```
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
[Step 2/11] Loading OpenVINO Runtime
[ INFO ] OpenVINO:
[ INFO ] Build ................................. 2024.3.0-1-1e3b88e4e3f
[ INFO ]
[ INFO ] Device info:
[ INFO ] GPU
[ INFO ] Build ................................. 2024.3.0-1-1e3b88e4e3f
[ INFO ]
[ INFO ]
[Step 3/11] Setting device configuration
[Step 4/11] Reading model files
[ INFO ] Loading model files
[ INFO ] Read model took 8.46 ms
[ INFO ] Original model I/O parameters:
[ INFO ] Network inputs:
[ INFO ]     input (node: input) : f32 / [N,C,D,H,W] / [1,3,16,224,224]
[ INFO ] Network outputs:
[ INFO ]     output (node: output) : f32 / [...] / [1,100]
[Step 5/11] Resizing model to match image sizes and given batch
[Step 6/11] Configuring input of the model
[ INFO ] Model batch size: 1
[ INFO ] Network inputs:
[ INFO ]     input (node: input) : f32 / [N,C,D,H,W] / [1,3,16,224,224]
[ INFO ] Network outputs:
[ INFO ]     output (node: output) : f32 / [...] / [1,100]
[Step 7/11] Loading the model to the device
[ INFO ] Compile model took 1932.68 ms
[Step 8/11] Querying optimal runtime parameters
[ INFO ] Model:
[ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 4
[ INFO ]   NETWORK_NAME: torch-jit-export
[ INFO ]   EXECUTION_DEVICES: GPU.1
[ INFO ]   AUTO_BATCH_TIMEOUT: 1000
[Step 9/11] Creating infer requests and preparing input tensors
[ WARNING ] No input files were given: all inputs will be filled with random values!
[ INFO ] Test Config 0
[ INFO ] input  ([N,C,D,H,W], f32, [1,3,16,224,224], static):	random (binary data/numpy array is expected)
[Step 10/11] Measuring performance (Start inference asynchronously, 4 inference requests, limits: 60000 ms duration)
[ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
[ INFO ] First inference took 6.32 ms
[Step 11/11] Dumping statistics report
[ INFO ] Execution Devices: [ GPU.1 ]
[ INFO ] Count:               28580 iterations
[ INFO ] Duration:            60009.64 ms
[ INFO ] Latency:
[ INFO ]    Median:           8.24 ms
[ INFO ]    Average:          8.36 ms
[ INFO ]    Min:              3.27 ms
[ INFO ]    Max:              13.66 ms
[ INFO ] Throughput:          476.26 FPS
```
