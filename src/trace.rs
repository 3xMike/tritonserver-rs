//! Tracing utilities for debugging and profiling.
//!
//! Usage example:
//! ```
//! struct TraceH;
//! impl TraceHandler for TraceH {
//!     fn trace_activity(
//!        &self,
//!        trace: &tritonserver_rs::trace::Trace,
//!        event: Activity,
//!        event_time: Duration,
//!     ) {
//!         log::info!(
//!             "Tracing activities: Trace_id: {}, event: {event:?}, event_time_secs: {}",
//!             trace.id().unwrap(),
//!             event_time.as_secs()
//!         );
//!         if event == Activity::ComputeStart {
//!             log::info!("Computations start, spawning new Trace");
//!             trace.spawn_child().unwrap();
//!         }
//!     }
//! }
//!
//! impl TensorTraceHandler for TraceH {
//!     fn trace_tensor_activity(
//!         &self,
//!         trace: &Trace,
//!         event: Activity,
//!         _tensor_data: &tritonserver_rs::Buffer,
//!         tensor_shape: tritonserver_rs::message::Shape,
//!     ) {
//!         log::info!(
//!             "Tracing Tensor Activity: Trace_id: {}, event: {event:?}, tensor name: {}",
//!             trace.id().unwrap(),
//!             tensor_shape.name
//!         );
//!     }
//! }
//!
//! /// Adds custom tracing to Inference Request.
//! fn add_trace_to_request(request: &mut Request) {
//!    request.add_trace(Trace::new_with_handle(
//!        Level::TIMESTAMPS | Level::TENSORS,
//!        0,
//!        TraceH,
//!        Some(TraceH),
//!    ).unwrap());
//! }
//! ```

use core::slice;
use std::{
    ffi::{c_void, CStr},
    mem::{forget, transmute},
    os::raw::c_char,
    ptr::{null, null_mut},
    sync::Arc,
    time::Duration,
};

use crate::{
    error::{Error, CSTR_CONVERT_ERROR_PLUG},
    from_char_array,
    message::Shape,
    sys, to_cstring, Buffer, MemoryType,
};

bitflags::bitflags! {
    /// Trace levels. The trace level controls the type of trace
    ///  activities that are reported for an inference request.
    ///
    /// Trace level values can be combined to trace multiple types of activities. For example, use
    /// ([Level::TIMESTAMPS] | [Level::TENSORS]) to trace both timestamps and
    ///  tensors for an inference request.
    struct Level: u32 {
        /// Tracing disabled. No trace activities are reported.
        const DISABLED = sys::tritonserver_tracelevel_enum_TRITONSERVER_TRACE_LEVEL_DISABLED;
        /// Deprecated. Use [Level::TIMESTAMPS].
        #[deprecated]
        const MIN = sys::tritonserver_tracelevel_enum_TRITONSERVER_TRACE_LEVEL_MIN;
        /// Deprecated. Use [Level::TIMESTAMPS].
        #[deprecated]
        const MAX = sys::tritonserver_tracelevel_enum_TRITONSERVER_TRACE_LEVEL_MAX;
        /// Record timestamps for the inference request.
        const TIMESTAMPS = sys::tritonserver_tracelevel_enum_TRITONSERVER_TRACE_LEVEL_TIMESTAMPS;
        /// Record input and output tensor values for the inference request.
        const TENSORS = sys::tritonserver_tracelevel_enum_TRITONSERVER_TRACE_LEVEL_TENSORS;
    }
}

impl Level {
    #[allow(dead_code)]
    /// Get the string representation of a trace level.
    fn as_str(self) -> &'static str {
        unsafe {
            let ptr = sys::TRITONSERVER_InferenceTraceLevelString(self.bits());
            assert!(!ptr.is_null());
            CStr::from_ptr(ptr)
                .to_str()
                .unwrap_or(CSTR_CONVERT_ERROR_PLUG)
        }
    }
}

/// Enum representation of inference status.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Activity {
    RequestStart = sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_REQUEST_START,
    QueueStart = sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_QUEUE_START,
    ComputeStart = sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_COMPUTE_START,
    ComputeInputEnd = sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_COMPUTE_INPUT_END,
    ComputeOutputStart =
        sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_COMPUTE_OUTPUT_START,
    ComputeEnd = sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_COMPUTE_END,
    RequestEnd = sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_REQUEST_END,
    TensorQueueInput = sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_TENSOR_QUEUE_INPUT,
    TensorBackendInput =
        sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_TENSOR_BACKEND_INPUT,
    TensorBackendOutput =
        sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_TENSOR_BACKEND_OUTPUT,
    CustomActivity = sys::tritonserver_traceactivity_enum_TRITONSERVER_TRACE_CUSTOM_ACTIVITY,
}

/// Inference event handler trait.
pub trait TraceHandler: Send + Sync + 'static {
    /// This function is invoked each time the `event` occures.
    ///
    /// `trace`: Trace object that was reported.
    /// Note that child traces of constructed one also are reported with this fn.
    /// Check [Trace::new_with_handle] for more info.\
    /// `event`: activity that has occurred. \
    /// `event_time`: time when event occured. \
    ///     Triton Trace APIs report timestamps using steady clock, which is a monotonic clock that ensures time always movess forward.
    ///     This clock is not related to wall clock time and, for example, can measure time since last reboot (aka /proc/uptime).
    fn trace_activity(&self, trace: &Trace, event: Activity, event_time: Duration);
}

impl TraceHandler for () {
    fn trace_activity(&self, _trace: &Trace, _event: Activity, _event_time: Duration) {}
}

/// Tensor event handler trait.
pub trait TensorTraceHandler: Send + Sync + 'static {
    /// This function is invoked each time the tensor `event` occures.
    ///
    /// `trace`: Trace object that was reported.
    /// Note that child traces of constructed one also are reported with this fn.
    /// Check [Trace::new_with_handle] for more info.\
    /// `event`: activity that has occurred. \
    /// `tensor_data`: borrowed buffer containing data of the tensor. \
    /// `tensor_shape`: shape (name, data_type and dims) of the tensor.
    fn trace_tensor_activity(
        &self,
        trace: &Trace,
        event: Activity,
        tensor_data: &Buffer,
        tensor_shape: Shape,
    );
}

impl TensorTraceHandler for () {
    fn trace_tensor_activity(
        &self,
        _trace: &Trace,
        _event: Activity,
        _tensor_data: &Buffer,
        _tensor_shape: Shape,
    ) {
    }
}

/// Can be passed to [Trace::new_with_handle] if no TENSORS or TIMESTAMPS are needed.
pub const NOOP: Option<()> = None;

struct TraceCallbackItems<H: TraceHandler, T: TensorTraceHandler> {
    activity_handler: Option<H>,
    tensor_activity_handler: Option<T>,
}

/// Don't want to use annotations like Trace<H, T> for
/// handlers_copy: Arc<TraceCallbackItems<H,T>>, so will use Arc<dyn DynamicTypeHelper>.
///
/// If someone can teach me how to do it better, i'm all ears((.
trait DynamicTypeHelper: Send + Sync {}
impl<H: TraceHandler, T: TensorTraceHandler> DynamicTypeHelper for TraceCallbackItems<H, T> {}

/// Inference object that provides custom tracing.
///
/// Is constructed with [TraceHandler] object that is activated each time an event occures.
pub struct Trace {
    pub(crate) ptr: TraceInner,
    /// So callback won't be dropped if trace reports after the fn delete (inference).
    handlers_copy: Arc<dyn DynamicTypeHelper>,
}

pub(crate) struct TraceInner(pub(crate) *mut sys::TRITONSERVER_InferenceTrace);
unsafe impl Send for TraceInner {}
unsafe impl Sync for TraceInner {}

impl PartialEq for Trace {
    fn eq(&self, other: &Self) -> bool {
        let left = match self.id() {
            Ok(l) => l,
            Err(err) => {
                log::warn!("Error getting ID for two Traces comparison: {err}");
                return false;
            }
        };
        let right = match other.id() {
            Ok(r) => r,
            Err(err) => {
                log::warn!("Error getting ID for two Traces comparison: {err}");
                return false;
            }
        };
        left == right
    }
}
impl Eq for Trace {}

impl Trace {
    /// Create a new inference trace object.
    ///
    /// The `activity_handler` and `tensor_activity_handler` will be called to report activity
    /// including [Trace::report_activity] called by this trace as well as by __every__ child traces that are spawned
    /// by this one. So the [TraceHandler::trace_activity] and [TensorTraceHandler::trace_tensor_activity]
    /// should check the trace object (first argument) that are passed to it
    /// to determine specifically what trace was reported.
    ///
    /// `level`: The tracing level. \
    /// `parent_id`: The parent trace id for this trace.
    /// A value of 0 indicates that there is not parent trace. \
    /// `activity_handler`: The callback function where activity (on timeline event)
    ///  for the trace (and all the child traces) is reported. \
    /// `tensor_activity_handler`: Optional callback function where activity (on tensor event)
    /// for the trace (and all the child traces) is reported.
    pub fn new_with_handle<H: TraceHandler, T: TensorTraceHandler>(
        parent_id: u64,
        activity_handler: Option<H>,
        tensor_activity_handler: Option<T>,
    ) -> Result<Self, Error> {
        let enable_activity = activity_handler.is_some();
        let enable_tensor_activity = tensor_activity_handler.is_some();

        let level = match (enable_activity, enable_tensor_activity) {
            (true, true) => Level::TENSORS | Level::TIMESTAMPS,
            (true, false) => Level::TIMESTAMPS,
            (false, true) => Level::TENSORS,
            (false, false) => Level::DISABLED,
        };

        let mut ptr = null_mut::<sys::TRITONSERVER_InferenceTrace>();
        let handlers = Arc::new(TraceCallbackItems {
            activity_handler,
            tensor_activity_handler,
        });
        let raw_handlers = Arc::into_raw(handlers.clone()) as *mut c_void;

        triton_call!(sys::TRITONSERVER_InferenceTraceTensorNew(
            &mut ptr as *mut _,
            level.bits(),
            parent_id,
            enable_activity.then_some(activity_wraper::<H, T>),
            enable_tensor_activity.then_some(tensor_activity_wrapper::<H, T>),
            Some(delete::<H, T>),
            raw_handlers,
        ))?;

        assert!(!ptr.is_null());
        let trace = Trace {
            ptr: TraceInner(ptr),
            handlers_copy: handlers,
        };
        Ok(trace)
    }

    /// Report a trace activity. All the traces reported using this API will be send [Activity::CustomActivity] type.
    ///
    /// `timestamp` The timestamp associated with the trace activity. \
    /// `name` The trace activity name.
    pub fn report_activity<N: AsRef<str>>(
        &self,
        timestamp: Duration,
        activity_name: N,
    ) -> Result<(), Error> {
        let name = to_cstring(activity_name)?;
        triton_call!(sys::TRITONSERVER_InferenceTraceReportActivity(
            self.ptr.0,
            timestamp.as_nanos() as _,
            name.as_ptr()
        ))
    }

    /// Get the id associated with the trace.
    /// Every trace is assigned an id that is unique across all traces created for a Triton server.
    pub fn id(&self) -> Result<u64, Error> {
        let mut id: u64 = 0;
        triton_call!(
            sys::TRITONSERVER_InferenceTraceId(self.ptr.0, &mut id as *mut _),
            id
        )
    }

    /// Get the parent id associated with the trace. \
    /// The parent id indicates a parent-child relationship between two traces.
    /// A parent id value of 0 indicates that there is no parent trace.
    pub fn parent_id(&self) -> Result<u64, Error> {
        let mut id: u64 = 0;
        triton_call!(
            sys::TRITONSERVER_InferenceTraceParentId(self.ptr.0, &mut id as *mut _),
            id
        )
    }

    /// Get the name of the model associated with the trace.
    pub fn model_name(&self) -> Result<String, Error> {
        let mut name = null::<c_char>();
        triton_call!(
            sys::TRITONSERVER_InferenceTraceModelName(self.ptr.0, &mut name as *mut _),
            from_char_array(name)
        )
    }

    /// Get the version of the model associated with the trace.
    pub fn model_version(&self) -> Result<i64, Error> {
        let mut version: i64 = 0;
        triton_call!(
            sys::TRITONSERVER_InferenceTraceModelVersion(self.ptr.0, &mut version as *mut _),
            version
        )
    }

    /// Get the request id associated with a trace.
    /// Returns the version of the model associated with the trace.
    pub fn request_id(&self) -> Result<String, Error> {
        let mut request_id = null::<c_char>();

        triton_call!(
            sys::TRITONSERVER_InferenceTraceRequestId(self.ptr.0, &mut request_id as *mut _),
            from_char_array(request_id)
        )
    }

    /// Returns the child trace, spawned from the parent(self) trace.
    ///
    /// Be causious: Trace is deleting on drop, so don't forget to save it.
    /// Also do not use parent and child traces for different Requests: it can lead to Seq Faults.
    pub fn spawn_child(&self) -> Result<Trace, Error> {
        let mut trace = null_mut();
        triton_call!(
            sys::TRITONSERVER_InferenceTraceSpawnChildTrace(self.ptr.0, &mut trace),
            Trace {
                ptr: TraceInner(trace),
                handlers_copy: self.handlers_copy.clone(),
            }
        )
    }

    /// Set context to Triton Trace.
    pub fn set_context(&mut self, context: String) -> Result<&mut Self, Error> {
        let context = to_cstring(context)?;
        triton_call!(
            sys::TRITONSERVER_InferenceTraceSetContext(self.ptr.0, context.as_ptr()),
            self
        )
    }

    /// Get Triton Trace context.
    pub fn context(&self) -> Result<String, Error> {
        let mut context = null::<c_char>();
        triton_call!(
            sys::TRITONSERVER_InferenceTraceContext(self.ptr.0, &mut context as *mut _),
            from_char_array(context)
        )
    }
}

impl Drop for TraceInner {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                sys::TRITONSERVER_InferenceTraceDelete(self.0);
            }
        }
    }
}

unsafe extern "C" fn delete<H: TraceHandler, T: TensorTraceHandler>(
    this: *mut sys::TRITONSERVER_InferenceTrace,
    userp: *mut c_void,
) {
    if !userp.is_null() && !this.is_null() {
        sys::TRITONSERVER_InferenceTraceDelete(this);
        Arc::from_raw(userp as *const TraceCallbackItems<H, T>);
    }
}

unsafe extern "C" fn activity_wraper<H: TraceHandler, T: TensorTraceHandler>(
    trace: *mut sys::TRITONSERVER_InferenceTrace,
    activity: sys::TRITONSERVER_InferenceTraceActivity,
    timestamp_ns: u64,
    userp: *mut ::std::os::raw::c_void,
) {
    if !userp.is_null() {
        let handle = Arc::from_raw(userp as *const TraceCallbackItems<H, T>);
        let foo_trace = Trace {
            ptr: TraceInner(trace),
            handlers_copy: handle.clone(),
        };
        let activity: Activity = transmute(activity);

        let timestamp = Duration::from_nanos(timestamp_ns);

        if let Some(activity_handle) = handle.activity_handler.as_ref() {
            activity_handle.trace_activity(&foo_trace, activity, timestamp)
        };

        // Drop will be in delete method.
        forget(handle);
        forget(foo_trace.ptr);
    }
}

unsafe extern "C" fn tensor_activity_wrapper<H: TraceHandler, T: TensorTraceHandler>(
    trace: *mut sys::TRITONSERVER_InferenceTrace,
    activity: sys::TRITONSERVER_InferenceTraceActivity,
    name: *const ::std::os::raw::c_char,
    datatype: sys::TRITONSERVER_DataType,
    base: *const ::std::os::raw::c_void,
    byte_size: usize,
    shape: *const i64,
    dim_count: u64,
    memory_type: sys::TRITONSERVER_MemoryType,
    _memory_type_id: i64,
    userp: *mut ::std::os::raw::c_void,
) {
    if !userp.is_null() {
        let handle = Arc::from_raw(userp as *const TraceCallbackItems<H, T>);

        let foo_trace = Trace {
            ptr: TraceInner(trace),
            handlers_copy: handle.clone(),
        };
        let activity: Activity = transmute(activity);

        let data_type = unsafe { transmute::<u32, crate::memory::DataType>(datatype) };
        let memory_type: MemoryType = unsafe { transmute(memory_type) };

        let tensor_shape = Shape {
            name: from_char_array(name),
            datatype: data_type,
            dims: slice::from_raw_parts(shape, dim_count as _).to_vec(),
        };

        let tensor_data = Buffer {
            ptr: base as *mut _,
            len: byte_size,
            data_type,
            memory_type,
            owned: false,
        };

        if let Some(tensor_activity_handler) = handle.tensor_activity_handler.as_ref() {
            tensor_activity_handler.trace_tensor_activity(
                &foo_trace,
                activity,
                &tensor_data,
                tensor_shape,
            )
        };

        forget(handle);
        forget(foo_trace.ptr);
        // Drop will be in delete method.
    }
}
