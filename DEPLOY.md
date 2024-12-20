# **Build**  

To build an application using **Tritonserver-rs**, the following requirements must be met:  
- **Triton library**  
- **APT packages**: `curl`, `build-essential`, `gdb`, `libclang-dev`, and `git`  
- **Rust**  
- **CUDA library** (required unless running in CPU-only mode)  

In addition, **Triton backends** are required to **run** the application.  

## Recommended Development Environment  

It is highly recommended to use a development environment based on the [official Triton Inference Server Docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver). The provided [Dockerfile.dev](./Dockerfile.dev) includes all necessary dependencies for building and running **Tritonserver-rs** applications.  

```sh
$ docker build --network host -t triton_dev:24.07 --build-arg TRITON_CONTAINER_VERSION=24.07 -f ../Dockerfile.dev .
```

You can find an example deployment using this container in the [examples folder](./examples/README.md).  

### Minimal TritonInferenceServer container version.
Since Triton C-lib API must not be older than our bindings API (1.25 currently), minimal TRITON_CONTAINER_VERSION is 24.07.

---

# **Production**  

One drawback of the development approach is the large size of the resulting Docker image. For production use, it is recommended to compile the application in the development container and then run it in a smaller [base image](./Dockerfile.base).  

## Minimizing Image Size  

To further reduce image size, you can create a minimal base image instead of using `nvcr.io/nvidia/tritonserver:<yy.mm>-py3`. For example, a minimal container can be built if only TensorRT and PyTorch backends are required. Refer to the [Triton Inference Server Customization Guide](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/compose.md) for more details.  

### Example Script for Minimal Base Image  

```sh
export TRITON_CONTAINER_VERSION=24.07
export MIN_IMG=min_img:$TRITON_CONTAINER_VERSION
export BASE_IMG=base:$TRITON_CONTAINER_VERSION

# Clone the Triton server repository and generate a minimal image
git clone https://github.com/triton-inference-server/server tmp/server && cd tmp/server

./compose.py --output-name $MIN_IMG \
   --backend pytorch --backend tensorrt --container-version $TRITON_CONTAINER_VERSION

cd ../../ && rm -rf tmp

# Build the base image
docker build --network host -t $BASE_IMG --build-arg TRITON_IMAGE=$MIN_IMG -f ./Dockerfile.base .
```  

---

# **CPU-Only Mode**  

To build **Tritonserver-rs** on platforms without GPU support, disable the `gpu` feature in your `Cargo.toml` file:  

```toml
[dependencies]
tritonserver-rs = { version = "0.1", default-features = false }
```  

## Important Notes  

If the system has a CUDA driver installed, **Triton** will attempt to use the GPU for inference by default. To enforce CPU-only inference, consider the following options:  

1. **Docker/Kubernetes Environment**:  
    Set the environment variable `NVIDIA_DRIVER_CAPABILITIES=graphics`. This will prevent the container from accessing the GPU.  

2. **Model Configuration**:  
    Define instance groups in the model configuration file to specify CPU-only inference:  
    ```pb
    instance_group [{ kind: KIND_CPU }]
    ```  
    Refer to the [Triton Model Configuration Guide](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups) for more details.  

3. **Server Options in Code**:  
    Set the following options in your application to disable GPU and pinned memory:  
    ```rust
    let mut options = tritonserver_rs::Options::new("path/to/models")?;
    options
        .pinned_memory_pool_byte_size(0)?
        .cuda_memory_pool_byte_size(0)?;
    ```  