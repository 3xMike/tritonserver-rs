pub(crate) mod infer;
mod utils;
pub use crate::trace::Trace;
pub use infer::{InferenceError, InputRelease, ResponseFuture};

use std::{collections::HashMap, ffi::CStr, os::raw::c_char, ptr::null, time::Duration};

use bitflags::bitflags;

use crate::{
    error::{ErrorCode, CSTR_CONVERT_ERROR_PLUG},
    memory::{Buffer, DataType, MemoryType},
    message::Shape,
    run_in_context, sys, to_cstring, Error, Server,
};

bitflags! {
    /// Inference request sequence flag.
    pub struct Sequence: u32 {
        const START = sys::tritonserver_requestflag_enum_TRITONSERVER_REQUEST_FLAG_SEQUENCE_START;
        const END = sys::tritonserver_requestflag_enum_TRITONSERVER_REQUEST_FLAG_SEQUENCE_END;
    }
}

/// Allocator, that user provides in order to allocate output buffers when they are needed for Triton. \
/// [Allocator::allocate] will be invoked after [Request::infer_async] call once for each model's output. The name of the requested output, it's memory type
/// and byte size are passed as arguments.
///
/// Allocator should be able to allocate buffer for each model's output.
///
/// **Note**: Allocated buffers can be returned via [crate::Response::return_buffers] method. \
/// **Note** allocate() method can be not invoked at all, for example, if model error occures before output is needed.
#[async_trait::async_trait]
pub trait Allocator: Send {
    /// Allocate output buffer for output with name `tensor_name`.
    ///
    /// **NOTES:**:
    /// - It's not necessary to allocate buffer on exact requested_memory_type: for example, it's fine to allocate buffer on Pinned when Triton requested GPU buffer.
    ///   The only requirement is not to allocate CPU memory when GPU is requested and vice versa.
    /// - Buffer of greater or equal size than requested `byte_size` can be allocated but not the smaller.
    /// - Allocated buffer's datatype must match this output datatype specified in the model's config.
    /// - Method will be invoked in asynchronous context.
    async fn allocate(
        &mut self,
        tensor_name: String,
        requested_memory_type: MemoryType,
        byte_size: usize,
    ) -> Result<Buffer, Error>;
}

/// Default allocator.
///
/// Will allocate exact `byte_size` bytes (datatype &ndash `u8`) of `requested_memory_type` for each output.
///
/// If tensor types provided via [add_tensor_type](DefaultAllocator::add_tensor_type) or [with_tensors_types](DefaultAllocator::with_tensors_types)
/// corresponding datatypes will be used for each output (instead of `u8`).
///
/// Note that if you use methods above, all the models outputs must be provided as hints.
/// If only few of models output is provided via methods above,
/// the DefaultAllocator::allocate will return [ErrorCode::NotFound] error on missing outputs.
#[derive(Debug, Default, Clone)]
pub struct DefaultAllocator {
    tensor_types: HashMap<String, DataType>,
}

impl DefaultAllocator {
    /// Create new DefaultAllocator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add output tensor type hint. I.e. Buffer of `data_type` data type will be allocated for output tensor `tensor_name`.
    ///
    /// Note that if you use this method, all the models outputs must be provided as hints.
    /// If only few of models output is provided,
    /// the DefaultAllocator::allocate will return [ErrorCode::NotFound] error on missing outputs.
    pub fn add_tensor_type<N: AsRef<str>>(&mut self, tensor_name: N, data_type: DataType) {
        let _ = self
            .tensor_types
            .insert(tensor_name.as_ref().to_string(), data_type);
    }

    /// Create DefaultAllocator with output tensors data type hints `tensor_types`.
    ///
    /// Note that if you use this method, all the models outputs must be provided as hints.
    /// If only few of models output is provided,
    /// the DefaultAllocator::allocate will return [ErrorCode::NotFound] error on missing outputs.
    pub fn with_tensors_types(tensor_types: HashMap<String, DataType>) -> Self {
        Self { tensor_types }
    }
}

#[async_trait::async_trait]
impl Allocator for DefaultAllocator {
    async fn allocate(
        &mut self,
        tensor_name: String,
        requested_mem_type: MemoryType,
        byte_size: usize,
    ) -> Result<Buffer, Error> {
        if self.tensor_types.is_empty() {
            run_in_context!(0, Buffer::alloc::<u8>(byte_size, requested_mem_type))
        } else {
            let data_type = *self.tensor_types.get(&tensor_name).ok_or_else(|| {
                Error::new(
                    ErrorCode::NotFound,
                    format!(
                        "Tensor {tensor_name} not found for DefaultAllocator: {:?}",
                        self.tensor_types
                    ),
                )
            })?;
            let data_type_size = data_type.size();
            run_in_context!(
                0,
                Buffer::alloc_with_data_type(
                    (byte_size as f32 / data_type_size as f32).ceil() as usize,
                    requested_mem_type,
                    data_type,
                )
            )
        }
    }
}

/// Inference request object.\
/// One can get this item using [Server::create_request].
///
/// It's required to add input data and Allocator to this structure before the inference via one of [add_input](Request::add_input) methods  via [Request::add_allocator] or [Request::add_default_allocator] method.
pub struct Request<'a> {
    ptr: *mut sys::TRITONSERVER_InferenceRequest,
    model_name: String,
    input: HashMap<String, Buffer>,
    custom_allocator: Option<Box<dyn Allocator>>,
    custom_trace: Option<Trace>,
    // Уверяемся, что Server не дропнется во время выполнения Request. \
    // Server(Arc<Inner>)
    server: &'a Server,
}

impl<'a> Request<'a> {
    pub(crate) fn new<M: AsRef<str>>(
        ptr: *mut sys::TRITONSERVER_InferenceRequest,
        server: &'a Server,
        model: M,
    ) -> Result<Request, Error> {
        Ok(Request {
            ptr,
            model_name: model.as_ref().to_string(),
            input: HashMap::new(),
            custom_allocator: None,
            custom_trace: None,
            server,
        })
    }

    /// Add custom Allocator to the request. \
    /// Check [Allocator] trait for more info.
    pub fn add_allocator(&mut self, custom_allocator: Box<dyn Allocator>) -> &mut Self {
        let _ = self.custom_allocator.replace(custom_allocator);
        self
    }

    /// Add [DefaultAllocator] to the request. \
    /// Check [Allocator] trait and [DefaultAllocator] for more info.
    pub fn add_default_allocator(&mut self) -> &mut Self {
        let _ = self
            .custom_allocator
            .replace(Box::new(DefaultAllocator::new()));
        self
    }

    /// Add custom Trace to the request. \
    /// If this method is not invoked, no tracing will be provided. \
    /// Check [Trace] for more info about tracing.
    pub fn add_trace(&mut self, custom_trace: Trace) -> &mut Self {
        let _ = self.custom_trace.replace(custom_trace);
        self
    }

    /// Get the ID of the request.
    pub fn get_id(&self) -> Result<&str, Error> {
        let mut id = null::<c_char>();
        triton_call!(sys::TRITONSERVER_InferenceRequestId(
            self.ptr,
            &mut id as *mut _
        ))?;

        assert!(!id.is_null());
        Ok(unsafe { CStr::from_ptr(id) }
            .to_str()
            .unwrap_or(CSTR_CONVERT_ERROR_PLUG))
    }

    /// Set the ID of the request.
    pub fn set_id<I: AsRef<str>>(&mut self, id: I) -> Result<&mut Self, Error> {
        let id = to_cstring(id)?;
        triton_call!(
            sys::TRITONSERVER_InferenceRequestSetId(self.ptr, id.as_ptr()),
            self
        )
    }

    /// Get the flag(s) associated with the request. \
    /// Check [Sequence] for available flags.
    pub fn get_flags(&self) -> Result<Sequence, Error> {
        let mut flag: u32 = 0;
        triton_call!(
            sys::TRITONSERVER_InferenceRequestFlags(self.ptr, &mut flag as *mut _),
            unsafe { Sequence::from_bits_unchecked(flag) }
        )
    }

    /// Set the flag(s) associated with a request. \
    /// Check [Sequence] for available flags.
    pub fn set_flags(&mut self, flags: Sequence) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_InferenceRequestSetFlags(self.ptr, flags.bits()),
            self
        )
    }

    /// Get the correlation ID of the inference request. \
    /// Default is 0, which indicates that the request has no correlation ID. \
    /// If the correlation id associated with the inference request is a string, this function will return a failure. \
    /// The correlation ID is used to indicate two or more inference request are related to each other. \
    /// How this relationship is handled by the inference server is determined by the model's scheduling policy.
    pub fn get_correlation_id(&self) -> Result<u64, Error> {
        let mut id: u64 = 0;
        triton_call!(
            sys::TRITONSERVER_InferenceRequestCorrelationId(self.ptr, &mut id as *mut _),
            id
        )
    }

    /// Get the correlation ID of the inference request as a string. \
    /// Default is empty "", which indicates that the request has no correlation ID. \
    /// If the correlation id associated with the inference request is an unsigned integer, then this function will return a failure. \
    /// The correlation ID is used to indicate two or more inference request are related to each other. \
    /// How this relationship is handled by the inference server is determined by the model's scheduling policy.
    pub fn get_correlation_id_as_str(&self) -> Result<&str, Error> {
        let mut id = null::<c_char>();
        triton_call!(sys::TRITONSERVER_InferenceRequestCorrelationIdString(
            self.ptr,
            &mut id as *mut _
        ))?;

        assert!(!id.is_null());
        Ok(unsafe { CStr::from_ptr(id) }
            .to_str()
            .unwrap_or(CSTR_CONVERT_ERROR_PLUG))
    }

    /// Set the correlation ID of the inference request to be an unsigned integer. \
    /// Default is 0, which indicates that the request has no correlation ID. \
    /// The correlation ID is used to indicate two or more inference request are related to each other. \
    /// How this relationship is handled by the inference server is determined by the model's scheduling policy.
    pub fn set_correlation_id(&mut self, id: u64) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_InferenceRequestSetCorrelationId(self.ptr, id),
            self
        )
    }

    /// Set the correlation ID of the inference request to be a string. \
    /// The correlation ID is used to indicate two or more inference request are related to each other. \
    /// How this relationship is handled by the inference server is determined by the model's scheduling policy.
    pub fn set_correlation_id_as_str<I: AsRef<str>>(&mut self, id: I) -> Result<&mut Self, Error> {
        let id = to_cstring(id)?;
        triton_call!(
            sys::TRITONSERVER_InferenceRequestSetCorrelationIdString(self.ptr, id.as_ptr()),
            self
        )
    }

    /// Get the priority of the request. \
    /// The default is 0 indicating that the request does not specify a priority and so will use the model's default priority.
    pub fn get_priority(&self) -> Result<u32, Error> {
        let mut priority: u32 = 0;
        triton_call!(
            sys::TRITONSERVER_InferenceRequestPriority(self.ptr, &mut priority as *mut _),
            priority
        )
    }

    /// Set the priority of the request. \
    /// The default is 0 indicating that the request does not specify a priority and so will use the model's default priority.
    pub fn set_priority(&mut self, priority: u32) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_InferenceRequestSetPriority(self.ptr, priority),
            self
        )
    }

    /// Get the timeout of the request. \
    /// The default is 0 which indicates that the request has no timeout.
    pub fn get_timeout(&self) -> Result<Duration, Error> {
        let mut timeout_us: u64 = 0;
        triton_call!(
            sys::TRITONSERVER_InferenceRequestTimeoutMicroseconds(
                self.ptr,
                &mut timeout_us as *mut _,
            ),
            Duration::from_micros(timeout_us)
        )
    }

    /// Set the timeout of the request. \
    /// The default is 0 which indicates that the request has no timeout.
    pub fn set_timeout(&mut self, timeout: Duration) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
                self.ptr,
                timeout.as_micros() as u64,
            ),
            self
        )
    }

    /// Add an input to the request.\
    /// `input_name`: The name of the input. \
    /// `buffer`: input data containing buffer. \
    /// Note: input data will be returned after the inference. Check [ResponseFuture::get_input_release] for more info.
    pub fn add_input<N: AsRef<str>>(
        &mut self,
        input_name: N,
        buffer: Buffer,
    ) -> Result<&mut Self, Error> {
        self.add_input_inner(input_name, buffer, None::<String>, None::<Vec<i64>>)
    }

    /// Add an input with the specified shape to the request.\
    /// `input_name`: The name of the input. \
    /// `buffer`: input data containing buffer. \
    ///  `dims`: Dimensions of the input.\
    /// Note: input data will be returned after the inference. Check [ResponseFuture::get_input_release] for more info.
    pub fn add_input_with_dims<N, D>(
        &mut self,
        input_name: N,
        buffer: Buffer,
        dims: D,
    ) -> Result<&mut Self, Error>
    where
        N: AsRef<str>,
        D: AsRef<[i64]>,
    {
        self.add_input_inner(input_name, buffer, None::<String>, Some(dims))
    }

    /// Add an input with the specified host policy to the request.\
    /// `input_name`: The name of the input.\
    /// `buffer`: input data containing buffer. \
    /// `policy`: The policy name, all model instances executing with this policy will use this input buffer for execution.\
    /// Note: input data will be returned after the inference. Check [ResponseFuture::get_input_release] for more info.
    pub fn add_input_with_policy<N, P>(
        &mut self,
        input_name: N,
        buffer: Buffer,
        policy: P,
    ) -> Result<&mut Self, Error>
    where
        N: AsRef<str>,
        P: AsRef<str>,
    {
        self.add_input_inner(input_name, buffer, Some(policy), None::<Vec<i64>>)
    }

    /// Add an input with the specified host policy and shape to the request.
    /// `input_name`: The name of the input.\
    /// `buffer`: input data containing buffer. \
    /// `policy`: The policy name, all model instances executing with this policy will use this input buffer for execution.\
    /// `dims`: Dimensions of the input.\
    /// Note: input data will be returned after the inference. Check [ResponseFuture::get_input_release] for more info.
    pub fn add_input_with_policy_and_dims<N, P, D>(
        &mut self,
        input_name: N,
        buffer: Buffer,
        policy: P,
        dims: D,
    ) -> Result<&mut Self, Error>
    where
        N: AsRef<str>,
        P: AsRef<str>,
        D: AsRef<[i64]>,
    {
        self.add_input_inner(input_name, buffer, Some(policy), Some(dims))
    }

    fn add_input_inner<N, P, D>(
        &mut self,
        input_name: N,
        buffer: Buffer,
        policy: Option<P>,
        dims: Option<D>,
    ) -> Result<&mut Self, Error>
    where
        N: AsRef<str>,
        P: AsRef<str>,
        D: AsRef<[i64]>,
    {
        if self.input.contains_key(input_name.as_ref()) {
            return Err(Error::new(
                ErrorCode::Alreadyxists,
                format!(
                    "Request already has buffer for input \"{}\"",
                    input_name.as_ref()
                ),
            ));
        }
        let model_shape = self.get_shape(input_name.as_ref())?;

        let shape = if let Some(dims) = dims {
            Shape {
                name: input_name.as_ref().to_string(),
                datatype: model_shape.datatype,
                dims: dims.as_ref().to_vec(),
            }
        } else {
            Shape {
                name: input_name.as_ref().to_string(),
                datatype: model_shape.datatype,
                dims: model_shape.dims.clone(),
            }
        };

        assert_buffer_shape(&shape, &buffer, input_name.as_ref())?;

        self.add_input_triton(&input_name, &shape)?;

        if let Some(policy) = policy {
            self.append_input_data_with_policy(input_name, &policy, buffer)?;
        } else {
            self.append_input_data(input_name, buffer)?;
        }

        Ok(self)
    }

    fn get_shape<N: AsRef<str>>(&self, source: N) -> Result<&Shape, Error> {
        let model_name = &self.model_name;
        let model = self.server.get_model(model_name)?;

        match model
            .inputs
            .iter()
            .find(|shape| shape.name == source.as_ref())
        {
            None => Err(Error::new(
                ErrorCode::Internal,
                format!("Model {model_name} has no input named: {}", source.as_ref()),
            )),
            Some(shape) => Ok(shape),
        }
    }

    fn add_input_triton<I: AsRef<str>>(&self, input_name: I, input: &Shape) -> Result<(), Error> {
        let name = to_cstring(input_name)?;
        triton_call!(sys::TRITONSERVER_InferenceRequestAddInput(
            self.ptr,
            name.as_ptr(),
            input.datatype as u32,
            input.dims.as_ptr(),
            input.dims.len() as u64,
        ))
    }

    fn append_input_data<I: AsRef<str>>(
        &mut self,
        input_name: I,
        buffer: Buffer,
    ) -> Result<&mut Self, Error> {
        let name = to_cstring(&input_name)?;
        triton_call!(sys::TRITONSERVER_InferenceRequestAppendInputData(
            self.ptr,
            name.as_ptr(),
            buffer.ptr,
            buffer.len,
            buffer.memory_type as u32,
            0,
        ))?;

        let _ = self.input.insert(input_name.as_ref().to_string(), buffer);
        Ok(self)
    }

    fn append_input_data_with_policy<I: AsRef<str>, P: AsRef<str>>(
        &mut self,
        input_name: I,
        policy: P,
        buffer: Buffer,
    ) -> Result<&mut Self, Error> {
        let name = to_cstring(&input_name)?;
        let policy = to_cstring(policy)?;
        triton_call!(
            sys::TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                self.ptr,
                name.as_ptr(),
                buffer.ptr,
                buffer.len,
                buffer.memory_type as u32,
                0,
                policy.as_ptr(),
            )
        )?;

        self.input.insert(input_name.as_ref().to_string(), buffer);
        Ok(self)
    }

    pub(crate) fn add_outputs(&mut self) -> Result<&mut Self, Error> {
        let model = self.server.get_model(&self.model_name)?;

        for output in &model.outputs {
            self.add_output(&output.name)?;
        }

        Ok(self)
    }

    /// Add an output request to an inference request.\
    /// name: The name of the output.\
    /// buffer: output data buffer that required by triton allocator.
    /// Embeddings will be put in this buffer.
    /// One can obtain buffer back using Response::output() or with infer_async() Error.
    pub(crate) fn add_output<N: AsRef<str>>(&mut self, name: N) -> Result<&mut Self, Error> {
        let output_name = to_cstring(&name)?;
        triton_call!(
            sys::TRITONSERVER_InferenceRequestAddRequestedOutput(self.ptr, output_name.as_ptr()),
            self
        )
    }
}

unsafe impl<'a> Send for Request<'a> {}

impl<'a> Drop for Request<'a> {
    fn drop(&mut self) {
        unsafe {
            sys::TRITONSERVER_InferenceRequestDelete(self.ptr);
        }
    }
}

fn assert_buffer_shape<N: AsRef<str>>(
    shape: &Shape,
    buffer: &Buffer,
    source: N,
) -> Result<(), Error> {
    if shape.datatype != buffer.data_type {
        return Err(Error::new(
            ErrorCode::InvalidArg,
            format!(
                "input buffer datatype {:?} missmatches model shape datatype: {:?}. input name: {}",
                buffer.data_type,
                shape.datatype,
                source.as_ref()
            ),
        ));
    }
    let shape_size =
        shape.dims.iter().filter(|n| **n > 0).product::<i64>() as u32 * shape.datatype.size();

    if shape_size as usize > buffer.size() {
        Err(Error::new(
            ErrorCode::InvalidArg,
            format!(
                "Buffer has size: {}, that less than shape min size: {shape_size}. input name: {}",
                buffer.size(),
                source.as_ref()
            ),
        ))
    } else {
        Ok(())
    }
}
