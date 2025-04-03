use std::{
    error::Error as ErrorExt,
    ffi::{CStr, CString},
    fmt, io,
    mem::transmute,
};

use crate::sys;

pub(crate) const CSTR_CONVERT_ERROR_PLUG: &str = "INVALID UTF-8 STRING";

/// Triton server error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ErrorCode {
    Unknown = sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_UNKNOWN,
    Internal = sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
    NotFound = sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_NOT_FOUND,
    InvalidArg = sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INVALID_ARG,
    Unavailable = sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_UNAVAILABLE,
    Unsupported = sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_UNSUPPORTED,
    Alreadyxists = sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_ALREADY_EXISTS,
}

/// Triton server error.
pub struct Error {
    pub(crate) ptr: *mut sys::TRITONSERVER_Error,
    pub(crate) owned: bool,
}

/// It's protected by the owned, so until no one changes owned it's safe.
/// User can't change it anyhow: it's private + pub methods don't change it.
unsafe impl Send for Error {}
unsafe impl Sync for Error {}

impl Error {
    /// Create new custom error.
    pub fn new<S: AsRef<str>>(code: ErrorCode, message: S) -> Self {
        let message = CString::new(message.as_ref()).expect("CString::new failed");
        unsafe {
            let this = sys::TRITONSERVER_ErrorNew(code as u32, message.as_ptr());
            assert!(!this.is_null());
            this.into()
        }
    }

    /// Return ErrorCode of the error.
    pub fn code(&self) -> ErrorCode {
        unsafe { transmute(sys::TRITONSERVER_ErrorCode(self.ptr)) }
    }

    /// Return string representation of the ErrorCode.
    pub fn name(&self) -> &str {
        let ptr = unsafe { sys::TRITONSERVER_ErrorCodeString(self.ptr) };
        if ptr.is_null() {
            "NULL"
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_str()
                .unwrap_or(CSTR_CONVERT_ERROR_PLUG)
        }
    }

    /// Return error description.
    pub fn message(&self) -> &str {
        let ptr = unsafe { sys::TRITONSERVER_ErrorMessage(self.ptr) };
        if ptr.is_null() {
            "NULL"
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_str()
                .unwrap_or(CSTR_CONVERT_ERROR_PLUG)
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub(crate) fn wrong_type(mem_type: crate::memory::MemoryType) -> Self {
        Self::new(
            ErrorCode::InvalidArg,
            format!("Got {mem_type:?} with gpu feature disabled"),
        )
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name(), self.message())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name(), self.message())
    }
}

impl From<*mut sys::TRITONSERVER_Error> for Error {
    fn from(ptr: *mut sys::TRITONSERVER_Error) -> Self {
        Error { ptr, owned: true }
    }
}

impl ErrorExt for Error {}

impl Drop for Error {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe { sys::TRITONSERVER_ErrorDelete(self.ptr) };
        }
    }
}

impl From<Error> for io::Error {
    fn from(err: Error) -> Self {
        io::Error::new(io::ErrorKind::Other, err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create() {
        const ERROR_CODE: ErrorCode = ErrorCode::Unknown;
        const ERROR_DESCRIPTION: &str = "some error";

        let err = Error::new(ERROR_CODE, ERROR_DESCRIPTION);

        assert_eq!(err.code(), ERROR_CODE);
        assert_eq!(err.message(), ERROR_DESCRIPTION);
    }
}
