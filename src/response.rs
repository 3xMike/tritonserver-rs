#![allow(clippy::arc_with_non_send_sync)]

use std::{
    collections::HashMap,
    ffi::{c_void, CStr},
    hint,
    mem::transmute,
    os::raw::c_char,
    ptr::{null, null_mut},
    slice::from_raw_parts,
    sync::Arc,
};

use log::trace;
use tokio::runtime::Handle;

use crate::{
    allocator::Allocator,
    error::{Error, CSTR_CONVERT_ERROR_PLUG},
    from_char_array,
    memory::{Buffer, DataType, MemoryType},
    parameter::{Parameter, ParameterContent},
    request::infer::InferenceError,
    sys,
};

/// Output tensor of the model.
///
/// Must not outlive the parent Response.
///
/// Each output is a reference on a part of
/// the output buffer (passed to request via Allocator) that contains the embedding.
/// May be smaller than initial buffer, if Triton does not need whole buffer.
#[derive(Debug)]
pub struct Output {
    /// Name of the output tensor.
    pub name: String,
    /// Shape (dims) of the output tensor.
    pub shape: Vec<i64>,
    buffer: Buffer,
    parent_response: Arc<InferenceResponseWrapper>,
    index_in_parent_response: u32,
}

// Can't copy Output and use it's ptr directly from public, so safe.
unsafe impl Send for Output {}
unsafe impl Sync for Output {}

impl Output {
    /// Get the Buffer containing the inference result (embedding).
    ///
    /// # Safety
    /// Do not mutate data of the returned value.
    /// If mutable (owned) Buffer is needed, use [Response::return_buffers].
    pub fn get_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get memory type of the output tensor.
    pub fn memory_type(&self) -> MemoryType {
        self.buffer.memory_type
    }

    /// Get data type of the output tensor.
    pub fn data_type(&self) -> DataType {
        self.buffer.data_type
    }

    /// Get a classification label associated with the output.
    pub fn classification_label(&self, class: u64) -> Result<String, Error> {
        self.parent_response
            .classification_label(self.index_in_parent_response, class)
    }
}

pub struct Response {
    outputs: Vec<Output>,
    triton_ptr_wrapper: Arc<InferenceResponseWrapper>,
    buffers_count: u32,
    /// Алокатор нужен тут, так как после вызова InferenceResponseWrapper::drop() тритон начинает вызывать
    /// release(), в которых участвует алокатор. Соответсвенно, он не должен быть уничтожен до этого момента.
    allocator: Arc<Allocator>,
    parameters: Vec<Parameter>,
}

unsafe impl Send for Response {}
unsafe impl Sync for Response {}

impl Response {
    /// Read the inference result, obtain output.
    pub(crate) fn new(
        ptr: *mut sys::TRITONSERVER_InferenceResponse,
        buffers_count: u32,
        allocator: Arc<Allocator>,
        runtime: Handle,
    ) -> Result<Self, InferenceError> {
        trace!("Response::new() is called");
        let wrapper = Arc::new(InferenceResponseWrapper(ptr));

        // Ошибка в ходе выполнения.
        if let Some(error) = wrapper.error() {
            drop(wrapper);

            if allocator.is_alloc_called() {
                // Waiting for the end of the release

                while allocator
                    .0
                    .returned_buffers
                    .load(std::sync::atomic::Ordering::Relaxed)
                    < buffers_count
                {
                    hint::spin_loop()
                }
            }

            let bufs = std::thread::spawn(move || {
                runtime.block_on(async move {
                    let mut bufs = allocator.0.output_buffers.write().await;
                    bufs.drain().collect()
                })
            })
            .join()
            .unwrap();

            return Err(InferenceError {
                error,
                output_buffers: bufs,
            });
        }

        let output_count = wrapper.output_count()?;

        if output_count != buffers_count {
            log::error!(
                "output_count: {output_count} != count of assigned output buffers: {buffers_count}",
            );
        }

        let mut outputs = Vec::new();
        let mut output_ids = Vec::new();
        trace!("Response::new() obtaining outputs");
        for output_id in 0..output_count {
            let output = wrapper.output(output_id)?;
            output_ids.push(output.name.clone());
            outputs.push(output);
        }

        let mut parameters = Vec::new();
        for parameter_id in 0..wrapper.parameter_count()? {
            parameters.push(wrapper.parameter(parameter_id)?);
        }

        Ok(Self {
            outputs,
            triton_ptr_wrapper: wrapper,
            buffers_count,
            allocator,
            parameters,
        })
    }

    /// The results of the inference.
    pub fn get_outputs(&self) -> &[Output] {
        &self.outputs
    }

    /// Get `output_name` result of the inference.
    pub fn get_output<O: AsRef<str>>(&self, output_name: O) -> Option<&Output> {
        self.outputs.iter().find(|o| o.name == output_name.as_ref())
    }

    /// Deconstruct the Response and get all the allocated output buffers back. \
    /// If you want just an immutable result of the inference, use [Response::get_outputs] or [Response::get_output] method.
    pub async fn return_buffers(self) -> Result<HashMap<String, Buffer>, Error> {
        // Triron will call `allocator::release()`
        // (therefore, we can get output buffer back)
        // ONLY after we call sys::TRITONSERVER_InferenceResponseDelete(),
        // that is the wrapper destructor.
        // each Output has Arc on wrapper so drop outputs first.
        drop(self.outputs);
        drop(self.triton_ptr_wrapper);
        trace!("return_buffer() awaiting on output receivers");
        let buffers_count = self.buffers_count;

        while self
            .allocator
            .0
            .returned_buffers
            .load(std::sync::atomic::Ordering::Relaxed)
            < buffers_count
        {
            hint::spin_loop()
        }

        let res = {
            let mut bufs = self.allocator.0.output_buffers.write().await;
            bufs.drain().collect()
        };

        drop(self.allocator);
        Ok(res)
    }

    /// Get model name and version used to produce thr response.
    pub fn model(&self) -> Result<(&str, i64), Error> {
        self.triton_ptr_wrapper.model()
    }

    /// Get the ID of the request corresponding to the response.
    pub fn id(&self) -> Result<String, Error> {
        self.triton_ptr_wrapper.id()
    }

    /// Get all information about the response parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        self.parameters.clone()
    }
}

#[derive(Debug)]
struct InferenceResponseWrapper(*mut sys::TRITONSERVER_InferenceResponse);

// Если в какой-то момент нужно будет вернуть все эти методы в публичное пространство, необходимо
// поставить lifetime на Output и Parameter.
impl InferenceResponseWrapper {
    /// Return the error status of an inference response.
    /// Return a Some(Error) object on failure, return None on success.
    fn error(&self) -> Option<Error> {
        let err = unsafe { sys::TRITONSERVER_InferenceResponseError(self.0) };
        if err.is_null() {
            None
        } else {
            Some(Error {
                ptr: err,
                owned: false,
            })
        }
    }

    /// Get model name and version used to produce a response.
    fn model(&self) -> Result<(&str, i64), Error> {
        let mut name = null::<c_char>();
        let mut version: i64 = 0;
        triton_call!(sys::TRITONSERVER_InferenceResponseModel(
            self.0,
            &mut name as *mut _,
            &mut version as *mut _,
        ))?;

        assert!(!name.is_null());
        Ok((
            unsafe { CStr::from_ptr(name) }
                .to_str()
                .unwrap_or(CSTR_CONVERT_ERROR_PLUG),
            version,
        ))
    }

    /// Get the ID of the request corresponding to a response.
    fn id(&self) -> Result<String, Error> {
        let mut id = null::<c_char>();
        triton_call!(
            sys::TRITONSERVER_InferenceResponseId(self.0, &mut id as *mut _),
            from_char_array(id)
        )
    }

    /// Get the number of parameters available in the response.
    fn parameter_count(&self) -> Result<u32, Error> {
        let mut count: u32 = 0;
        triton_call!(
            sys::TRITONSERVER_InferenceResponseParameterCount(self.0, &mut count as *mut _),
            count
        )
    }

    /// Get all information about a parameter.
    fn parameter(&self, index: u32) -> Result<Parameter, Error> {
        let mut name = null::<c_char>();
        let mut kind: sys::TRITONSERVER_ParameterType = 0;
        let mut value = null::<c_void>();
        triton_call!(sys::TRITONSERVER_InferenceResponseParameter(
            self.0,
            index,
            &mut name as *mut _,
            &mut kind as *mut _,
            &mut value as *mut _,
        ))?;

        assert!(!name.is_null());
        assert!(!value.is_null());
        let name = unsafe { CStr::from_ptr(name) }
            .to_str()
            .unwrap_or(CSTR_CONVERT_ERROR_PLUG);
        let value = match kind {
            sys::TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_STRING => {
                ParameterContent::String(
                    unsafe { CStr::from_ptr(value as *const c_char) }
                        .to_str()
                        .unwrap_or(CSTR_CONVERT_ERROR_PLUG)
                        .to_string(),
                )
            }
            sys::TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_INT => {
                ParameterContent::Int(unsafe { *(value as *mut i64) })
            }
            sys::TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_BOOL => {
                ParameterContent::Bool(unsafe { *(value as *mut bool) })
            }
            _ => unreachable!(),
        };
        Parameter::new(name, value)
    }

    /// Get the number of outputs available in the response.
    fn output_count(&self) -> Result<u32, Error> {
        let mut count: u32 = 0;
        triton_call!(
            sys::TRITONSERVER_InferenceResponseOutputCount(self.0, &mut count as *mut _),
            count
        )
    }

    fn output(self: &Arc<Self>, index: u32) -> Result<Output, Error> {
        let mut name = null::<c_char>();
        let mut data_type: sys::TRITONSERVER_DataType = 0;
        let mut shape = null::<i64>();
        let mut dim_count: u64 = 0;
        let mut base = null::<c_void>();
        let mut byte_size: libc::size_t = 0;
        let mut memory_type: sys::TRITONSERVER_MemoryType = 0;
        let mut memory_type_id: i64 = 0;
        let mut userp = null_mut::<c_void>();

        triton_call!(sys::TRITONSERVER_InferenceResponseOutput(
            self.0,
            index,
            &mut name as *mut _,
            &mut data_type as *mut _,
            &mut shape as *mut _,
            &mut dim_count as *mut _,
            &mut base as *mut _,
            &mut byte_size as *mut _,
            &mut memory_type as *mut _,
            &mut memory_type_id as *mut _,
            &mut userp as *mut _,
        ))?;

        assert!(!name.is_null());
        assert!(!base.is_null());

        let name = unsafe { CStr::from_ptr(name) }
            .to_str()
            .unwrap_or(CSTR_CONVERT_ERROR_PLUG)
            .to_string();

        let shape = if dim_count == 0 {
            log::trace!(
                "Model returned output '{name}' of shape []. Consider removing this output"
            );
            Vec::new()
        } else {
            unsafe { from_raw_parts(shape, dim_count as usize) }.to_vec()
        };
        let data_type = unsafe { transmute::<u32, crate::memory::DataType>(data_type) };
        let memory_type: MemoryType = unsafe { transmute(memory_type) };

        // Not owned buffer, because we can't move or mutate it,
        // we just borrow it from triton.
        let buffer = Buffer {
            ptr: base as *mut _,
            len: byte_size as usize,
            data_type,
            memory_type,
            owned: false,
        };
        Ok(Output {
            name,
            shape,
            buffer,
            index_in_parent_response: index,
            parent_response: self.clone(),
        })
    }

    /// Get a classification label associated with an output for a given index.
    fn classification_label(&self, index: u32, class: u64) -> Result<String, Error> {
        let mut label = null::<c_char>();
        triton_call!(
            sys::TRITONSERVER_InferenceResponseOutputClassificationLabel(
                self.0,
                index,
                class as usize,
                &mut label as *mut _,
            ),
            from_char_array(label)
        )
    }
}

impl Drop for InferenceResponseWrapper {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                sys::TRITONSERVER_InferenceResponseDelete(self.0);
            }
        }
    }
}
