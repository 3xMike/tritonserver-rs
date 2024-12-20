//! # **Perform easy and efficient ML model inference**
//!
//! This crate is designed to run **any** Machine Learning model on **any** architecture with ease and efficiency.  
//! It leverages the [Triton Inference Server](https://github.com/triton-inference-server/server)
//! (specifically the [Triton C library](https://github.com/triton-inference-server/core)) and provides a similar API with comparable advantages.
//! However, **Tritonserver-rs** allows you to build the inference server locally, offering significant performance benefits.
//! Check the [benchmark](https://github.com/3xMike/tritonserver-rs/blob/main/BENCH.md) for more details.
//!
//! ---
//!
//! # Usage  
//!
//! Run inference in three simple steps:
//!
//! ## **Step 1. Prepare the model repository**  
//!
//! Organize your model files in the following structure:
//!
//! ```text
//! models/
//! ├── yolov8/
//! |    ├── config.pbtxt
//! |    ├── 1/
//! |    │   └── model.onnx
//! |    ├── 2/
//! |    │   └── model.onnx
//! |    └── `<other versions of yolov8>`/
//! └── `<other models>`/
//! ```
//!
//! **Rules**:  
//! - All models must be stored in the same root directory (`models/` in this example).  
//! - Each model resides in its own folder containing:
//!   - A `config.pbtxt` configuration file.
//!   - One or more subdirectories, each representing a version of the model and containing the model file (e.g., `model.onnx`).  
//!
//! ---
//!
//! ## **Step 2. Write the code**  
//!
//! Add **Tritonserver-rs** to your `Cargo.toml`:  
//!
//! ```toml
//! [dependencies]
//! tritonserver-rs = "0.1"
//! ```
//!
//! Then write your application code:  
//!
//! ```rust
//! use tritonserver_rs::{Buffer, options::Options, Server};
//! use std::time::Duration;
//!
//! // Configure server options.
//! let mut opts = Options::new("models/")?;
//!
//! opts.exit_timeout(Duration::from_secs(5))?
//!     .backend_directory("/opt/tritonserver/backends")?;
//!
//! // Create the server.
//! let server = Server::new(opts).await?;
//!
//! // Input data.
//! let image = image::open("/data/cats.jpg")?;
//! let image = image.as_flat_samples_u8();
//!
//! // Create a request (specify the model name and version).
//! let mut request = server.create_request("yolov8", 2)?;
//!
//! // Add input data and an allocator.
//! request
//!     .add_default_allocator()
//!     .add_input("IMAGE", Buffer::from(image))?;
//!
//! // Run inference.
//! let fut = request.infer_async()?;
//!
//! // Obtain results.
//! let response = fut.await?;
//! ```
//!
//! ---
//!
//! ## **Step 3. Deploy**
//!
//! Here is an example of how to deploy using `docker-compose.yml`:  
//!
//! ```yml
//! my_app:
//!   image: {DEV_IMAGE}
//!   volumes:
//!     - ./Cargo.toml:/project/
//!     - ./src:/project/src
//!     - ../models:/models
//!     - ../cats.jpg:/data/cats.jpg
//!   entrypoint: ["cargo", "run", "--manifest-path=/project/Cargo.toml"]
//! ```
//!
//! We recommend using Dockerfile.dev as `{DEV_IMAGE}`. For more details on suitable images and deployment instructions, see DEPLOY.md.  
//!
//! ---
//!
//! # **More Information**
//!
//! For further details, check out the following resources (in [github repo](https://github.com/3xMike/tritonserver-rs/blob/main)):  
//! - [Examples](https://github.com/3xMike/tritonserver-rs/blob/main/examples): Learn how to run various ML models using **Tritonserver-rs**, configure inference, prepare models, and deploy.  
//! - [Model configuration guide](https://github.com/3xMike/tritonserver-rs/blob/main/MODEL_CONFIGURATION.md).  
//! - [Build and deployment instructions](https://github.com/3xMike/tritonserver-rs/blob/main/DEPLOY.md).  
//! - [Benchmark results](https://github.com/3xMike/tritonserver-rs/blob/main/BENCH.md).  
//! - [Triton Inference Server guides](https://github.com/triton-inference-server/server/tree/main/docs/README.md).  
//!
//! ---
//!
//! # **Advantages of the Crate**
//!
//! - **Versatility**: Extensive configuration options for models and servers.  
//! - **High performance**: Optimized for maximum efficiency.  
//! - **Broad backend support**: Run PyTorch, ONNX, TensorFlow, TensorRT, OpenVINO, model pipelines, and custom backends out of the box.  
//! - **Compatibility**: Supports most GPUs and architectures.  
//! - **Multi-model handling**: Handle multiple models simultaneously.  
//! - **Prometheus integration**: Built-in support for monitoring.  
//! - **CUDA-optimized**: Directly handle model inputs and outputs on GPU memory.  
//! - **Dynamic server management**: Advanced runtime control features.  
//! - **Rust-based**: Enjoy the safety, speed, and concurrency benefits of Rust.
//!
//! # Tritonserver C-lib API version
//! `1.33` (Minimal TRITON_CONTAINER_VERSION=23.07).

#![allow(clippy::bad_bit_mask)]

/// Macros to run some Cuda operations in context.
#[macro_use]
pub mod macros;

pub(crate) mod allocator;
#[cfg(feature = "gpu")]
/// Cuda context for managing device execution.
pub mod context;
/// Error types for Tritonserver-rs.
pub mod error;
/// Memory management utilities for model inference.
pub mod memory;
/// Metadata message serialization/deserialization.
pub mod message;
/// Performance metrics collection and reporting.
pub mod metrics;
/// Configuration options for Tritonserver-rs server.
pub mod options;
/// Model inference requests and server parameters.
pub mod parameter;
/// Request builder and utilities for Triton server inference.
pub mod request;
/// Response handling and parsing from Triton server.
pub mod response;
/// Server initialization and lifecycle management.
pub mod server;
pub(crate) mod sys {
    #![allow(
        non_camel_case_types,
        non_upper_case_globals,
        non_snake_case,
        dead_code,
        unused_imports,
        rustdoc::invalid_html_tags
    )]
    include!(concat!(env!("OUT_DIR"), "/tritonserver.rs"));
}
pub mod trace;

pub use crate::{
    error::{Error, ErrorCode},
    memory::{Buffer, MemoryType},
    request::{Allocator, Request},
    response::Response,
    server::Server,
    sys::{TRITONSERVER_API_VERSION_MAJOR, TRITONSERVER_API_VERSION_MINOR},
};
#[cfg(feature = "gpu")]
pub use context::{get_context, init_cuda};

use std::{
    ffi::{CStr, CString},
    os::{raw::c_char, unix::ffi::OsStrExt as _},
    path::Path,
};

/// Get the TRITONBACKEND API version supported by the Triton library.
/// This value can be compared against the TRITONSERVER_API_VERSION_MAJOR and TRITONSERVER_API_VERSION_MINOR used to build the client to ensure that Triton shared library is compatible with the client.
pub fn api_version() -> Result<(u32, u32), Error> {
    let mut major: u32 = 0;
    let mut minor: u32 = 0;

    triton_call!(
        sys::TRITONSERVER_ApiVersion(&mut major as *mut _, &mut minor as *mut _),
        (major, minor)
    )
}

pub(crate) fn to_cstring<S: AsRef<str>>(value: S) -> Result<CString, Error> {
    CString::new(value.as_ref().as_bytes())
        .map_err(|err| Error::new(ErrorCode::InvalidArg, format!("{}", err)))
}

pub(crate) fn path_to_cstring<P: AsRef<Path>>(value: P) -> Result<CString, Error> {
    value
        .as_ref()
        .canonicalize()
        .map_err(|err| Error::new(ErrorCode::InvalidArg, err.to_string()))
        .and_then(|path| {
            CString::new(path.as_os_str().as_bytes())
                .map_err(|err| Error::new(ErrorCode::InvalidArg, err.to_string()))
        })
}

pub(crate) fn from_char_array(value: *const c_char) -> String {
    assert!(!value.is_null());
    unsafe { CStr::from_ptr(value) }
        .to_str()
        .unwrap_or(error::CSTR_CONVERT_ERROR_PLUG)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api() {
        let (major, minor) = api_version().unwrap();

        assert_eq!(major, TRITONSERVER_API_VERSION_MAJOR);
        assert_eq!(minor, TRITONSERVER_API_VERSION_MINOR);
    }
}
