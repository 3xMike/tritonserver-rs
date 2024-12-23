use std::{
    collections::HashMap,
    ffi::c_void,
    ptr::null_mut,
    sync::{atomic::AtomicBool, Arc},
};

use log::trace;
use tokio::{
    runtime::Handle,
    sync::oneshot::{self, Receiver},
};

use crate::{
    allocator::Allocator,
    error::{Error, ErrorCode},
    memory::Buffer,
    sys, Request, Response,
};

/// Inference result error. Contains output buffers that was allocated by user provided Allocator during the inference.
#[derive(Debug)]
pub struct InferenceError {
    pub error: Error,
    pub output_buffers: HashMap<String, Buffer>,
}

impl From<Error> for InferenceError {
    fn from(error: Error) -> Self {
        Self {
            error,
            output_buffers: HashMap::new(),
        }
    }
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for InferenceError {}

/// Future that returns the inference response. \
/// The request can be cancelled by dropping this structure.
///
/// Also the input buffers assigned to the request can be returned via [get_input_release](ResponseFuture::get_input_release).
pub struct ResponseFuture {
    pub(super) response_receiver: Receiver<Result<Response, InferenceError>>,
    pub(super) input_release: Option<InputRelease>,
    pub(super) request_ptr: Arc<RequestCanceller>,
}

pub(super) struct RequestCanceller {
    pub(crate) is_inferenced: AtomicBool,
    pub(crate) request_ptr: *mut sys::TRITONSERVER_InferenceRequest,
}
unsafe impl Send for RequestCanceller {}
unsafe impl Sync for RequestCanceller {}

/// Struct that returns input buffers assigned to the request. \
/// Note: input buffer can be released in any time from the start of the inference
/// to the end of it.
///
/// Input buffers will be dropped if no one will await on this struct.
pub struct InputRelease(pub(super) oneshot::Receiver<HashMap<String, Buffer>>);

/// Start inference.
impl Request<'_> {
    /// Perform inference using the metadata and inputs supplied by the Request(self). \
    /// If the function returns success,
    /// the returned struct can be used to get results (.await) of the inference and
    /// to return input buffers after the inference start [ResponseFuture::get_input_release]. \
    /// Note: output buffer will be returned with [Response] or [InferenceError]. \
    pub fn infer_async(mut self) -> Result<ResponseFuture, Error> {
        // Check on all buffers are set.
        if self.input.is_empty() {
            return Err(Error::new(
                ErrorCode::NotFound,
                "Request's output buffer is not set",
            ));
        }
        if self.custom_allocator.is_none() {
            return Err(Error::new(
                ErrorCode::NotFound,
                "Request's output buffers allocator is not set",
            ));
        }
        let custom_allocator = self.custom_allocator.take().unwrap();
        let trace = self.custom_trace.take();

        // Add outputs.
        let datatype_hints = self.add_outputs()?;
        let outputs_count = self.server.get_model(&self.model_name)?.outputs.len();

        let runtime = self.server.runtime.clone();
        let request_ptr = self.ptr;
        let server_ptr = self.server.ptr.as_mut_ptr();

        // Канал, по которому мы вернем input buffer пользователю.
        let (input_tx, input_rx) = oneshot::channel();
        // На всякий случай сохраним указатель, в случае ошибки sys::TRITONSERVER_InferenceRequestSetReleaseCallback
        // разыменуем его и правильно дропнем Request.
        let boxed_request_input_recover = Box::into_raw(Box::new((self, input_tx)));
        let drop_boxed_request = |boxed_request: *mut (Request, _)| {
            let (_restored_request, _) = unsafe { *Box::from_raw(boxed_request) };
        };

        // Здесь мы отдаем Request, он нам вернется в методе release_callback.
        // Там же будет возвращен input_buffer.
        let err = unsafe {
            sys::TRITONSERVER_InferenceRequestSetReleaseCallback(
                request_ptr,
                Some(release_callback),
                boxed_request_input_recover as *mut _,
            )
        };

        if !err.is_null() {
            drop_boxed_request(boxed_request_input_recover);

            let err = Error {
                ptr: err,
                owned: true,
            };
            return Err(err);
        }

        // Allocator отправляется в alloc -> release, там он выдает запрашиваемые тритоном буферы в alloc и шлет их обратно в release.
        // Так как Allocator используется тритоном в методе release, который вызывается после удаления Response,
        // необходимо отправить алокатор в response_wrapper -> Response, чтобы Arc не дропнулся раньше времени.
        // Имена буферов отправляется в response_wrapper, на нем будем ждать возвращенные буферы для Response.
        let allocator = Arc::new(Allocator::new(
            custom_allocator,
            datatype_hints,
            runtime.clone(),
        )?);

        let allocator_ptr = Arc::as_ptr(&allocator);
        // response_tx отправляется в response_wrapper,
        // когда там сконструируется Response, он будет положен в tx.
        // response_rx отправляется юзеру внутри ResponseFuture, он на нем await-ится.
        let (response_tx, response_rx) = oneshot::channel();

        triton_call!(sys::TRITONSERVER_InferenceRequestSetResponseCallback(
            request_ptr,
            allocator.get_allocator(),
            allocator_ptr as *mut c_void,
            Some(responce_wrapper),
            Box::into_raw(Box::new(ResponseCallbackItems {
                response_tx,
                allocator,
                outputs_count,
                runtime,
            })) as *mut _,
        ))?;

        let trace_ptr = trace
            .as_ref()
            .map(|trace| trace.ptr.0)
            .unwrap_or_else(null_mut);

        triton_call!(sys::TRITONSERVER_ServerInferAsync(
            server_ptr,
            request_ptr,
            trace_ptr
        ))?;

        if let Some(trace) = trace {
            std::mem::forget(trace.ptr);
        }

        Ok(ResponseFuture {
            response_receiver: response_rx,
            input_release: Some(InputRelease(input_rx)),
            request_ptr: Arc::new(RequestCanceller {
                request_ptr,
                is_inferenced: AtomicBool::new(false),
            }),
        })
    }
}

struct ResponseCallbackItems {
    response_tx: oneshot::Sender<Result<Response, InferenceError>>,
    allocator: Arc<Allocator>,
    outputs_count: usize,
    runtime: Handle,
}

/// C-code returns the ownership on Request using this method.
unsafe extern "C" fn release_callback(
    ptr: *mut sys::TRITONSERVER_InferenceRequest,
    _flags: u32,
    user_data: *mut c_void,
) {
    trace!("release_callback is called");
    assert!(!ptr.is_null());
    assert!(!user_data.is_null());

    let (mut request, input_tx) = *Box::from_raw(user_data as *mut (Request, oneshot::Sender<_>));
    // Drain the input buffers
    let mut buffers = HashMap::new();
    std::mem::swap(&mut buffers, &mut request.input);

    if input_tx.send(buffers).is_err() {
        log::debug!("InputRelease was dropped before the input buffers returned from triton. Input buffers will be dropped");
    }

    assert_eq!(request.ptr, ptr);
    trace!("release_callback is ended");
}

/// C-code calls this method when Response is ready.
unsafe extern "C" fn responce_wrapper(
    response: *mut sys::TRITONSERVER_InferenceResponse,
    _flags: u32,
    user_data: *mut c_void,
) {
    trace!("response wrapper is called");
    assert!(!response.is_null());
    assert!(!user_data.is_null());

    // Allocator присылали сюда только для того, чтобы он не дропнулся во время реквеста.
    let ResponseCallbackItems {
        response_tx,
        allocator,
        outputs_count,
        runtime,
    } = *Box::from_raw(user_data as *mut ResponseCallbackItems);

    let send_res = response_tx.send(Response::new(
        response,
        outputs_count as u32,
        allocator,
        runtime,
    ));
    if send_res.is_err() {
        log::error!("error sending the result of the inference. It will be lost (including the output buffer)")
    } else {
        trace!("response wrapper: result is sent to oneshot");
    }
}
