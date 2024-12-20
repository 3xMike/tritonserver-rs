# **Perform easy and efficient ML model inference**

This crate is designed to run **any** Machine Learning model on **any** architecture with ease and efficiency.  
It leverages the [Triton Inference Server](https://github.com/triton-inference-server/server) (specifically the [Triton C library](https://github.com/triton-inference-server/core)) and provides a similar API with comparable advantages. However, **Tritonserver-rs** allows you to build the inference server locally, offering significant performance benefits. Check the [benchmark](./BENCH.md) for more details.

---

# Usage  

Run inference in three simple steps:

## **Step 1. Prepare the model repository**  

Organize your model files in the following structure:

```
models/
├── yolov8/
|    ├── config.pbtxt
|    ├── 1/
|    │   └── model.onnx
|    ├── 2/
|    │   └── model.onnx
|    └── `<other versions of yolov8>`/
└── `<other models>`/
```

**Rules**:  
- All models must be stored in the same root directory (`models/` in this example).  
- Each model resides in its own folder containing:
  - A `config.pbtxt` configuration file.
  - One or more subdirectories, each representing a version of the model and containing the model file (e.g., `model.onnx`).  

---

## **Step 2. Write the code**  

Add **Tritonserver-rs** to your `Cargo.toml`:  

```toml
[dependencies]
tritonserver-rs = "0.1"
```

Then write your application code:  

```rust
use tritonserver_rs::{Buffer, options::Options, Server};
use std::time::Duration;

// Configure server options.
let mut opts = Options::new("models/")?;

opts.exit_timeout(Duration::from_secs(5))?
    .backend_directory("/opt/tritonserver/backends")?;

// Create the server.
let server = Server::new(opts).await?;

// Input data.
let image = image::open("/data/cats.jpg")?;
let image = image.as_flat_samples_u8();

// Create a request (specify the model name and version).
let mut request = server.create_request("yolov8", 2)?;

// Add input data and an allocator.
request
    .add_default_allocator()
    .add_input("IMAGE", Buffer::from(image))?;

// Run inference.
let fut = request.infer_async()?;

// Obtain results.
let response = fut.await?;
```

---

## **Step 3. Deploy**

Here is an example of how to deploy using `docker-compose.yml`:  

```yml
my_app:
  image: {DEV_IMAGE}
  volumes:
    - ./Cargo.toml:/project/
    - ./src:/project/src
    - ../models:/models
    - ../cats.jpg:/data/cats.jpg
  entrypoint: ["cargo", "run", "--manifest-path=/project/Cargo.toml"]
```

We recommend using [Dockerfile.dev](./Dockerfile.dev) as `{DEV_IMAGE}`. For more details on suitable images and deployment instructions, see [DEPLOY.md](./DEPLOY.md).  

---

# **More Information**

For further details, check out the following resources:  
- [Examples](./examples/): Learn how to run various ML models using **Tritonserver-rs**, configure inference, prepare models, and deploy.  
- [Model configuration guide](MODEL_CONFIGURATION.md).  
- [Build and deployment instructions](DEPLOY.md).  
- [Benchmark results](BENCH.md).  
- [Triton Inference Server guides](https://github.com/triton-inference-server/server/tree/main/docs/README.md).  
- Documentation on [docs.rs](https://docs.rs/tritonserver-rs/).  

---

# **Advantages of the Crate**

- **Versatility**: Extensive configuration options for models and servers.  
- **High performance**: Optimized for maximum efficiency.  
- **Broad backend support**: Run PyTorch, ONNX, TensorFlow, TensorRT, OpenVINO, model pipelines, and custom backends out of the box.  
- **Compatibility**: Supports most GPUs and architectures.  
- **Multi-model handling**: Handle multiple models simultaneously.  
- **Prometheus integration**: Built-in support for monitoring.  
- **CUDA-optimized**: Directly handle model inputs and outputs on GPU memory.  
- **Dynamic server management**: Advanced runtime control features.  
- **Rust-based**: Enjoy the safety, speed, and concurrency benefits of Rust.

# Tritonserver C-lib API version
`1.33` (Minimal TRITON_CONTAINER_VERSION=23.07).