use std::{
    collections::HashMap,
    ffi::{c_void, CStr},
    mem::transmute,
    path::Path,
    ptr::null_mut,
    sync::Arc,
    time::Duration,
};

use serde_json::{from_slice, Value};

use crate::{
    message::{self, Index, Message, Model},
    metrics::{self, Metrics},
    options::Options,
    parameter::{Parameter, ParameterContent},
    path_to_cstring, sys, to_cstring, Error, ErrorCode, Request,
};

/// Batch properties of the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Batch {
    /// Triton cannot determine the batching properties of the model.
    /// This means that the model does not support batching in any way that is useable by Triton.
    Unknown = sys::tritonserver_batchflag_enum_TRITONSERVER_BATCH_UNKNOWN,
    /// The model supports batching along the first dimension of every input and output tensor.
    /// Triton schedulers that perform batching can automatically batch inference requests along this dimension.
    FirstDim = sys::tritonserver_batchflag_enum_TRITONSERVER_BATCH_FIRST_DIM,
}

/// Transaction policy of the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Transaction {
    /// The model generates exactly one response per request.
    OneToOne = sys::tritonserver_txn_property_flag_enum_TRITONSERVER_TXN_ONE_TO_ONE,
    /// The model may generate zero to many responses per request.
    Decoupled = sys::tritonserver_txn_property_flag_enum_TRITONSERVER_TXN_DECOUPLED,
}

bitflags::bitflags! {
    /// Flags that control how to collect the index.
    pub struct State: u32 {
        /// If set in 'flags', only the models that are loaded into the server and ready for inferencing are returned.
        const READY = sys::tritonserver_modelindexflag_enum_TRITONSERVER_INDEX_FLAG_READY;
    }
}

/// Kinds of instance groups recognized by TRITONSERVER.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum InstanceGroup {
    Auto = sys::TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_AUTO,
    Cpu = sys::TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_CPU,
    Gpu = sys::TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_GPU,
    Model = sys::TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_MODEL,
}

impl InstanceGroup {
    fn as_cstr(self) -> &'static CStr {
        unsafe { CStr::from_ptr(sys::TRITONSERVER_InstanceGroupKindString(self as u32)) }
    }

    /// Get the string representation of an instance-group kind.
    pub fn as_str(self) -> &'static str {
        self.as_cstr()
            .to_str()
            .unwrap_or(crate::error::CSTR_CONVERT_ERROR_PLUG)
    }
}

#[derive(Debug)]
pub(crate) struct Inner(*mut sys::TRITONSERVER_Server);
impl Inner {
    pub(crate) fn stop(&self) -> Result<(), Error> {
        triton_call!(sys::TRITONSERVER_ServerStop(self.0))
    }

    pub(crate) fn is_live(&self) -> Result<bool, Error> {
        let mut result = false;
        triton_call!(
            sys::TRITONSERVER_ServerIsLive(self.0, &mut result as *mut _),
            result
        )
    }

    pub(crate) fn delete(&self) -> Result<(), Error> {
        triton_call!(sys::TRITONSERVER_ServerDelete(self.0))
    }

    pub(crate) fn as_mut_ptr(&self) -> *mut sys::TRITONSERVER_Server {
        self.0
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let _ = self
            .is_live()
            .and_then(|live| {
                if live {
                    self.stop().and_then(|_| loop {
                        if !self.is_live()? {
                            return Ok(());
                        }
                    })
                } else {
                    Ok(())
                }
            })
            .and_then(|_| self.delete());
    }
}

/// # SAFETY
/// Inner is Send. But it's not Sync! \
/// However, it's used only in Server and Server is never clones Inner,
/// so there is always only 1 copy of it.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

/// Inference server object.
#[derive(Debug)]
pub struct Server {
    pub(crate) ptr: Arc<Inner>,
    pub(crate) models: HashMap<String, Model>,
    pub(crate) runtime: tokio::runtime::Handle,
}

unsafe impl Send for Server {}
impl Server {
    /// Create new server object.
    pub async fn new(options: Options) -> Result<Self, Error> {
        let mut server = null_mut::<sys::TRITONSERVER_Server>();
        triton_call!(sys::TRITONSERVER_ServerNew(
            &mut server as *mut _,
            options.0
        ))?;

        assert!(!server.is_null());

        let mut server = Server {
            ptr: Arc::new(Inner(server)),
            models: HashMap::new(),
            runtime: tokio::runtime::Handle::current(),
        };
        server.update_all_models()?;

        Ok(server)
    }

    pub(crate) fn get_model<M: AsRef<str>>(&self, model: M) -> Result<&Model, Error> {
        self.models.get(model.as_ref()).ok_or_else(|| {
            Error::new(
                ErrorCode::NotFound,
                format!(
                    "Model {} is not found in server model metadata storage.",
                    model.as_ref()
                ),
            )
        })
    }

    fn update_all_models(&mut self) -> Result<(), Error> {
        for model in self.model_index(State::all())? {
            self.update_model_info(model.name)?;
        }
        Ok(())
    }

    fn update_model_info<M: AsRef<str>>(&mut self, model: M) -> Result<(), Error> {
        self.models
            .insert(model.as_ref().to_string(), self.model_metadata(model, -1)?);
        Ok(())
    }

    /// Stop a server object. A server can't be restarted once it has been stopped.
    pub fn stop(&self) -> Result<(), Error> {
        self.ptr.stop()
    }

    /// Create a request to the model `model` of version `version`. \
    /// If version is set as `-1`, the server will choose a version based on the model's policy.
    pub fn create_request<M: AsRef<str>>(&self, model: M, version: i64) -> Result<Request, Error> {
        let model_name = to_cstring(model.as_ref())?;
        let mut ptr = null_mut::<sys::TRITONSERVER_InferenceRequest>();

        triton_call!(sys::TRITONSERVER_InferenceRequestNew(
            &mut ptr as *mut _,
            self.ptr.as_mut_ptr(),
            model_name.as_ptr(),
            version,
        ))?;

        assert!(!ptr.is_null());
        Request::new(ptr, self, model)
    }

    /// Check the model repository for changes and update server state based on those changes.
    pub fn poll_model_repository(&mut self) -> Result<(), Error> {
        triton_call!(sys::TRITONSERVER_ServerPollModelRepository(
            self.ptr.as_mut_ptr()
        ))?;

        self.update_all_models()
    }

    /// Set the exit timeout on the server object. This value overrides the value initially set through server options and provides a mechanism to update the exit timeout while the serving is running.
    ///
    /// `timeout` The exit timeout.
    pub fn set_exit_timeout(&mut self, timeout: Duration) -> Result<&mut Self, Error> {
        triton_call!(
            sys::TRITONSERVER_ServerSetExitTimeout(self.ptr.as_mut_ptr(), timeout.as_secs() as _),
            self
        )
    }

    /// Register a new model repository. Not available in polling mode.
    ///
    /// `repository` The full path to the model repository. \
    /// `name_mapping` List of name_mapping parameters.
    /// Each mapping has the model directory name as its key,
    /// overridden model name as its value.
    pub fn register_model_repo<P: AsRef<Path>, N: AsRef<str>>(
        &mut self,
        repository: P,
        name_mapping: HashMap<String, String>,
    ) -> Result<&mut Self, Error> {
        let path = path_to_cstring(repository)?;

        let mut mapping_params = name_mapping
            .into_iter()
            .map(|(k, v)| {
                Parameter::new(k, ParameterContent::String(v)).map(|param| param.ptr as *const _)
            })
            .collect::<Result<Vec<_>, _>>()?;

        triton_call!(
            sys::TRITONSERVER_ServerRegisterModelRepository(
                self.ptr.as_mut_ptr(),
                path.as_ptr(),
                mapping_params.as_mut_ptr(),
                mapping_params.len() as _
            ),
            self
        )
    }

    /// Unregister a model repository. Not available in polling mode.
    ///
    /// `repository_path` The full path to the model repository.
    pub fn unregister_model_repo<P: AsRef<Path>, N: AsRef<str>>(
        &mut self,
        repository: P,
    ) -> Result<&mut Self, Error> {
        let path = path_to_cstring(repository)?;

        triton_call!(
            sys::TRITONSERVER_ServerUnregisterModelRepository(self.ptr.as_mut_ptr(), path.as_ptr()),
            self
        )
    }

    /// Returns true if server is live, false otherwise.
    pub fn is_live(&self) -> Result<bool, Error> {
        self.ptr.is_live()
    }

    /// Returns true if server is ready, false otherwise.
    pub fn is_ready(&self) -> Result<bool, Error> {
        let mut result = false;

        triton_call!(
            sys::TRITONSERVER_ServerIsReady(self.ptr.as_mut_ptr(), &mut result as *mut _),
            result
        )
    }

    /// Returns true if the model is ready. \
    /// `name`: The name of the model to get readiness for. \
    /// `version`: The version of the model to get readiness for. If -1 then the server will choose a version based on the model's policy. \
    pub fn model_is_ready<N: AsRef<str>>(&self, name: N, version: i64) -> Result<bool, Error> {
        let name = to_cstring(name)?;
        let mut result = false;

        triton_call!(
            sys::TRITONSERVER_ServerModelIsReady(
                self.ptr.as_mut_ptr(),
                name.as_ptr(),
                version,
                &mut result as *mut _,
            ),
            result
        )
    }

    /// Get the batch properties of the model. \
    /// `name`: The name of the model. \
    /// `version`: The version of the model. If -1 then the server will choose a version based on the model's policy. \
    pub fn model_batch_properties<N: AsRef<str>>(
        &self,
        name: N,
        version: i64,
    ) -> Result<Batch, Error> {
        let name = to_cstring(name)?;
        let mut result: u32 = 0;
        let mut ptr = null_mut::<c_void>();

        triton_call!(sys::TRITONSERVER_ServerModelBatchProperties(
            self.ptr.as_mut_ptr(),
            name.as_ptr(),
            version,
            &mut result as *mut _,
            &mut ptr as *mut _,
        ))?;
        unsafe { Ok(transmute::<u32, Batch>(result)) }
    }

    /// Get the transaction policy of the model. \
    /// `name`: The name of the model. \
    /// `version`: The version of the model. If -1 then the server will choose a version based on the model's policy. \
    pub fn model_transaction_properties<N: AsRef<str>>(
        &self,
        name: N,
        version: i64,
    ) -> Result<Transaction, Error> {
        let name = to_cstring(name)?;
        let mut result: u32 = 0;
        let mut ptr = null_mut::<c_void>();

        triton_call!(sys::TRITONSERVER_ServerModelTransactionProperties(
            self.ptr.as_mut_ptr(),
            name.as_ptr(),
            version,
            &mut result as *mut _,
            &mut ptr as *mut _,
        ))?;
        unsafe { Ok(transmute::<u32, Transaction>(result)) }
    }

    /// Get the metadata of the server as a Message(json) object.
    pub fn metadata(&self) -> Result<message::Server, Error> {
        let mut result = null_mut::<sys::TRITONSERVER_Message>();

        triton_call!(sys::TRITONSERVER_ServerMetadata(
            self.ptr.as_mut_ptr(),
            &mut result as *mut _
        ))?;

        assert!(!result.is_null());
        Message(result).to_json().and_then(|json| {
            from_slice(json).map_err(|err| Error::new(ErrorCode::Internal, err.to_string()))
        })
    }

    /// Get the metadata of a model as a Message(json) object.\
    /// `name`: The name of the model. \
    /// `version`: The version of the model. If -1 then the server will choose a version based on the model's policy.
    pub fn model_metadata<N: AsRef<str>>(&self, name: N, version: i64) -> Result<Model, Error> {
        let name = to_cstring(name)?;
        let mut result = null_mut::<sys::TRITONSERVER_Message>();

        triton_call!(sys::TRITONSERVER_ServerModelMetadata(
            self.ptr.as_mut_ptr(),
            name.as_ptr(),
            version,
            &mut result as *mut _,
        ))?;

        assert!(!result.is_null());
        Message(result).to_json().and_then(|json| {
            from_slice(json).map_err(|err| Error::new(ErrorCode::Internal, err.to_string()))
        })
    }

    /// Get the statistics of a model as a Message(json) object. \
    /// `name`: The name of the model. \
    /// `version`: The version of the model. If -1 then the server will choose a version based on the model's policy.
    pub fn model_statistics<N: AsRef<str>>(&self, name: N, version: i64) -> Result<Value, Error> {
        let name = to_cstring(name)?;
        let mut result = null_mut::<sys::TRITONSERVER_Message>();

        triton_call!(sys::TRITONSERVER_ServerModelStatistics(
            self.ptr.as_mut_ptr(),
            name.as_ptr(),
            version,
            &mut result as *mut _,
        ))?;

        assert!(!result.is_null());
        Message(result).to_json().and_then(|json| {
            from_slice(json).map_err(|err| Error::new(ErrorCode::Internal, err.to_string()))
        })
    }

    /// Get the configuration of a model as a Message(json) object. \
    /// `name`: The name of the model. \
    /// `version`: The version of the model. If -1 then the server will choose a version based on the model's policy. \
    /// `config`: The model configuration will be returned in a format matching this version. \
    /// If the configuration cannot be represented in the requested version's format then an error will be returned.
    /// Currently only version 1 is supported.
    pub fn model_config<N: AsRef<str>>(
        &self,
        name: N,
        version: i64,
        config: u32,
    ) -> Result<Value, Error> {
        let name = to_cstring(name)?;
        let mut result = null_mut::<sys::TRITONSERVER_Message>();

        triton_call!(sys::TRITONSERVER_ServerModelConfig(
            self.ptr.as_mut_ptr(),
            name.as_ptr(),
            version,
            config,
            &mut result as *mut _,
        ))?;

        assert!(!result.is_null());
        Message(result).to_json().and_then(|json| {
            from_slice(json).map_err(|err| Error::new(ErrorCode::Internal, err.to_string()))
        })
    }

    /// Get the index of all unique models in the model repositories as a Message(json) object.
    pub fn model_index(&self, flags: State) -> Result<Vec<Index>, Error> {
        let mut result = null_mut::<sys::TRITONSERVER_Message>();

        triton_call!(sys::TRITONSERVER_ServerModelIndex(
            self.ptr.as_mut_ptr(),
            flags.bits(),
            &mut result as *mut _,
        ))?;

        assert!(!result.is_null());
        Message(result).to_json().and_then(|json| {
            from_slice(json).map_err(|err| Error::new(ErrorCode::Internal, err.to_string()))
        })
    }

    /// Load the requested model or reload the model if it is already loaded. \
    /// The function does not return until the model is loaded or fails to load \.
    /// `name`: The name of the model.
    pub fn load_model<N: AsRef<str>>(&mut self, name: N) -> Result<(), Error> {
        let model_name = to_cstring(&name)?;

        triton_call!(sys::TRITONSERVER_ServerLoadModel(
            self.ptr.as_mut_ptr(),
            model_name.as_ptr()
        ))?;

        self.update_model_info(name)
    }

    /// Load the requested model or reload the model if it is already loaded, with load parameters provided. \
    /// The function does not return until the model is loaded or fails to load. \
    /// Currently the below parameter names are recognized:
    ///
    /// - "config" : string parameter that contains a JSON representation of the
    ///   model configuration. This config will be used for loading the model instead
    ///   of the one in the model directory.
    ///
    /// Can be usefull if is needed to load the model with altered config.
    /// For example, if it's required to load only one exact version of the model (see [Parameter::from_config_with_exact_version] for more info).
    ///
    /// `name`: The name of the model. \
    /// `parameters`: slice of parameters.
    pub fn load_model_with_parametrs<N: AsRef<str>, P: AsRef<[Parameter]>>(
        &mut self,
        name: N,
        parameters: P,
    ) -> Result<(), Error> {
        let model_name = to_cstring(&name)?;
        let params_count = parameters.as_ref().len();
        let mut parametrs = parameters
            .as_ref()
            .iter()
            .map(|p| p.ptr.cast_const())
            .collect::<Vec<_>>();

        triton_call!(sys::TRITONSERVER_ServerLoadModelWithParameters(
            self.ptr.as_mut_ptr(),
            model_name.as_ptr(),
            parametrs.as_mut_ptr(),
            params_count as _,
        ))?;

        self.update_model_info(name)
    }

    /// Unload the requested model. \
    /// Unloading a model that is not loaded on server has no affect and success code will be returned. \
    /// The function does not wait for the requested model to be fully unload and success code will be returned. \
    /// `name`: The name of the model.
    pub fn unload_model<N: AsRef<str>>(&mut self, name: N) -> Result<(), Error> {
        let model_name = to_cstring(&name)?;

        triton_call!(sys::TRITONSERVER_ServerUnloadModel(
            self.ptr.as_mut_ptr(),
            model_name.as_ptr()
        ))?;

        self.update_model_info(name)
    }

    /// Unload the requested model, and also unload any dependent model that was loaded along with the requested model
    /// (for example, the models composing an ensemble). \
    /// Unloading a model that is not loaded on server has no affect and success code will be returned. \
    /// The function does not wait for the requested model and all dependent models to be fully unload and success code will be returned. \
    /// `name`: The name of the model.
    pub fn unload_model_and_dependents<N: AsRef<str>>(&mut self, name: N) -> Result<(), Error> {
        let model_name = to_cstring(&name)?;

        triton_call!(sys::TRITONSERVER_ServerUnloadModelAndDependents(
            self.ptr.as_mut_ptr(),
            model_name.as_ptr(),
        ))?;

        self.update_model_info(name)
    }

    /// Get the current metrics for the server.
    pub fn metrics(&self) -> Result<metrics::Metrics, Error> {
        let mut metrics = null_mut::<sys::TRITONSERVER_Metrics>();

        triton_call!(sys::TRITONSERVER_ServerMetrics(
            self.ptr.as_mut_ptr(),
            &mut metrics as *mut _
        ))?;

        assert!(!metrics.is_null());
        Ok(Metrics(metrics))
    }

    pub fn is_log_enabled(&self, level: LogLevel) -> bool {
        unsafe { sys::TRITONSERVER_LogIsEnabled(level as u32) }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum LogLevel {
    Info = sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_INFO,
    Warn = sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_WARN,
    Error = sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_ERROR,
    Verbose = sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_VERBOSE,
}
