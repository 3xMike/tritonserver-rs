use std::{fs::File, path::Path, ptr::null_mut};

use crate::{error::Error, sys, to_cstring};

/// Types of parameters recognized by TRITONSERVER.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum TritonParameterType {
    String = sys::TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_STRING,
    Int = sys::TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_INT,
    Bool = sys::TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_BOOL,
    Double = sys::TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_DOUBLE,
    Bytes = sys::TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_BYTES,
}

/// Enum representation of Parameter content.
#[derive(Debug, Clone)]
pub enum ParameterContent {
    String(String),
    Int(i64),
    Bool(bool),
    Double(f64),
    Bytes(Vec<u8>),
}

/// Parameter of the [Server](crate::Server) or [Response](crate::Response).
#[derive(Debug)]
pub struct Parameter {
    pub(crate) ptr: *mut sys::TRITONSERVER_Parameter,
    pub name: String,
    pub content: ParameterContent,
}

unsafe impl Send for Parameter {}

impl Parameter {
    /// Create new Parameter.
    pub fn new<N: AsRef<str>>(name: N, value: ParameterContent) -> Result<Self, Error> {
        let c_name = to_cstring(&name)?;
        let ptr = match &value {
            ParameterContent::Bool(v) => unsafe {
                sys::TRITONSERVER_ParameterNew(
                    c_name.as_ptr(),
                    TritonParameterType::Bool as _,
                    v as *const bool as *const _,
                )
            },
            ParameterContent::Int(v) => unsafe {
                sys::TRITONSERVER_ParameterNew(
                    c_name.as_ptr(),
                    TritonParameterType::Int as _,
                    v as *const i64 as *const _,
                )
            },
            ParameterContent::String(v) => {
                let v = to_cstring(v)?;
                unsafe {
                    sys::TRITONSERVER_ParameterNew(
                        c_name.as_ptr(),
                        TritonParameterType::String as _,
                        v.as_ptr() as *const _,
                    )
                }
            }
            ParameterContent::Double(v) => unsafe {
                sys::TRITONSERVER_ParameterNew(
                    c_name.as_ptr(),
                    TritonParameterType::Double as _,
                    v as *const f64 as *const _,
                )
            },
            ParameterContent::Bytes(v) => unsafe {
                sys::TRITONSERVER_ParameterBytesNew(
                    c_name.as_ptr(),
                    v.as_ptr() as *const _,
                    v.len() as _,
                )
            },
        };

        Ok(Self {
            ptr,
            name: name.as_ref().to_string(),
            content: value,
        })
    }

    /// Create String Parameter of model config with exact version of the model. \
    /// `config`: model config.pbtxt as json value.
    /// Check [load_config_as_json] to permutate .pbtxt config to json value. \
    /// If [Options::model_control_mode](crate::options::Options::model_control_mode) set as EXPLICIT and the result of this method is passed to [crate::Server::load_model_with_parametrs],
    /// the server will load only that exact model and only that exact version of it.
    pub fn from_config_with_exact_version(
        mut config: serde_json::Value,
        version: i64,
    ) -> Result<Self, Error> {
        config["version_policy"] = serde_json::json!({"specific": { "versions": [version]}});
        Parameter::new("config", ParameterContent::String(config.to_string()))
    }
}

impl Clone for Parameter {
    fn clone(&self) -> Self {
        Parameter::new(self.name.clone(), self.content.clone()).unwrap_or_else(|err| {
            log::warn!("Error cloning parameter: {err}. Result will be empty, do not use it.");
            Parameter {
                ptr: null_mut(),
                name: String::new(),
                content: ParameterContent::String(String::new()),
            }
        })
    }
}

impl Drop for Parameter {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::TRITONSERVER_ParameterDelete(self.ptr) }
        }
    }
}

fn hjson_to_json(value: serde_hjson::Value) -> serde_json::Value {
    match value {
        serde_hjson::Value::Null => serde_json::Value::Null,
        serde_hjson::Value::U64(v) => serde_json::Value::from(v),
        serde_hjson::Value::I64(v) => serde_json::Value::from(v),
        serde_hjson::Value::F64(v) => serde_json::Value::from(v),
        serde_hjson::Value::Bool(v) => serde_json::Value::from(v),
        serde_hjson::Value::String(v) => serde_json::Value::from(v),

        serde_hjson::Value::Array(v) => {
            serde_json::Value::from_iter(v.into_iter().map(hjson_to_json))
        }
        serde_hjson::Value::Object(v) => serde_json::Value::from_iter(
            v.into_iter()
                .map(|(key, value)| (key, hjson_to_json(value))),
        ),
    }
}

/// Load config.pbtxt from the `config_path` and parse it to json value. \
/// Might be useful if it is required to run model with altered config.
/// In this case String [Parameter] with name 'config' and the result of this method as data should be created
/// and passed to [Server::load_model_with_parametrs](crate::Server::load_model_with_parametrs) ([Options::model_control_mode](crate::options::Options::model_control_mode) set as EXPLICIT required).
/// Check realization of [Parameter::from_config_with_exact_version] as an example. \
/// **Note (Subject to change)**: congig must be in [hjson format](https://hjson.github.io/).
pub fn load_config_as_json<P: AsRef<Path>>(config_path: P) -> Result<serde_json::Value, Error> {
    let content = File::open(config_path).map_err(|err| {
        Error::new(
            crate::error::ErrorCode::InvalidArg,
            format!("Error opening the config file: {err}"),
        )
    })?;
    let value = serde_hjson::from_reader::<_, serde_hjson::Value>(&content).map_err(|err| {
        Error::new(
            crate::error::ErrorCode::InvalidArg,
            format!("Error parsing the config file as hjson: {err}"),
        )
    })?;
    Ok(hjson_to_json(value))
}

#[test]
fn test_config_to_json() {
    let json_cfg = serde_json::json!({
        "name": "voicenet",
        "platform": "onnxruntime_onnx",
        "input": [
            {
                "data_type": "TYPE_FP32",
                "name": "input",
                "dims": [512, 160000]
            }
        ],
        "output": [
            {
                "data_type": "TYPE_FP32",
                "name": "output",
                "dims": [512, 512]
            }
        ],
        "instance_group": [
            {
                "count": 2,
                "kind": "KIND_CPU"
            }
        ],
        "optimization": { "execution_accelerators": {
            "cpu_execution_accelerator" : [ {
                "name" : "openvino"
            } ]
        }}
    });

    assert_eq!(
        load_config_as_json("model_repo/voicenet_onnx/voicenet/config.pbtxt").unwrap(),
        json_cfg
    );
}
