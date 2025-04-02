use std::ptr::null;

use serde::{
    de::{Error as _, Unexpected},
    Deserialize, Deserializer,
};

use crate::{error::Error, memory::DataType, server::State, sys};

/// Representation of any configuration json message that server can send.
#[derive(Debug)]
pub(crate) struct Message(pub(crate) *mut sys::TRITONSERVER_Message);

impl Message {
    /// Get the serialized message in JSON format.
    pub(crate) fn to_json(&self) -> Result<&[u8], Error> {
        #[cfg(target_arch = "x86_64")]
        let mut ptr = null::<i8>();
        #[cfg(target_arch = "aarch64")]
        let mut ptr = null::<u8>();
        let mut size: usize = 0;

        triton_call!(sys::TRITONSERVER_MessageSerializeToJson(
            self.0,
            &mut ptr as *mut _,
            &mut size as *mut _,
        ))?;

        assert!(!ptr.is_null());
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const u8, size) })
    }
}

impl Drop for Message {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                sys::TRITONSERVER_MessageDelete(self.0);
            }
        }
    }
}

/// Loaded Model information.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize)]
pub struct Index {
    pub name: String,
    #[serde(deserialize_with = "de_state")]
    pub state: State,
    #[serde(deserialize_with = "de_version")]
    pub version: i64,
}

/// Model's metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize)]
pub struct Model {
    pub name: String,
    pub platform: String,
    pub versions: Vec<String>,
    pub inputs: Vec<Shape>,
    pub outputs: Vec<Shape>,
}

/// Shape of the tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize)]
pub struct Shape {
    pub name: String,
    #[serde(deserialize_with = "de_datatype")]
    pub datatype: DataType,
    #[serde(rename(deserialize = "shape"))]
    pub dims: Vec<i64>,
}

/// Server's metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize)]
pub struct Server {
    pub name: String,
    pub version: String,
    pub extensions: Vec<String>,
}

fn de_datatype<'de, D>(de: D) -> Result<DataType, D::Error>
where
    D: Deserializer<'de>,
{
    <&str>::deserialize(de)
        .and_then(|s| DataType::try_from(s).map_err(|_| D::Error::unknown_variant(s, &[])))
}

fn de_version<'de, D>(de: D) -> Result<i64, D::Error>
where
    D: Deserializer<'de>,
{
    <&str>::deserialize(de).and_then(|s| {
        s.parse::<i64>()
            .map_err(|_| D::Error::invalid_type(Unexpected::Str(s), &"i64"))
    })
}

fn de_state<'de, D>(de: D) -> Result<State, D::Error>
where
    D: Deserializer<'de>,
{
    <&str>::deserialize(de).and_then(|s| {
        if s == "READY" {
            Ok(State::READY)
        } else {
            Err(D::Error::invalid_value(Unexpected::Str(s), &"READY"))
        }
    })
}
