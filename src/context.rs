use std::{collections::HashMap, ffi::c_int, sync::Arc};

use cuda_driver_sys::{
    cuCtxCreate_v2, cuCtxDestroy_v2, cuCtxGetApiVersion, cuCtxPopCurrent_v2, cuCtxPushCurrent_v2,
    cuDeviceGet, cuDeviceGetAttribute, cuDeviceGetName, cuDeviceTotalMem_v2, cuInit, CUcontext,
    CUdevice, CUdevice_attribute,
};
use parking_lot::{Once, RwLock};

use crate::{error::Error, from_char_array};

/// Initialize Cuda runtime. Should be called before any Cuda function, perfectly &mdash; on the start of the application.
pub fn init_cuda() -> Result<(), Error> {
    cuda_call!(cuInit(0))
}

lazy_static::lazy_static! {
    static ref CUDA_CONTEXTS: RwLock<HashMap<i32, Arc<Context>>> = RwLock::new(HashMap::default());
    static ref ONCE: Once = Once::new();
}

/// Get Cuda context on device.
pub fn get_context(device: i32) -> Result<Arc<Context>, Error> {
    if let Some(ctx) = CUDA_CONTEXTS.read().get(&device) {
        return Ok(ctx.clone());
    }

    ONCE.call_once(|| init_cuda().unwrap());

    let dev = CuDevice::new(device)?;
    log::info!(
        "Using: {} {:.2}Gb",
        dev.get_name().unwrap(),
        dev.get_total_mem().unwrap() as f64 / (1_000_000_000) as f64
    );

    let arc = Arc::new(Context::new(dev, 0)?);
    CUDA_CONTEXTS.write().insert(device, arc.clone());

    Ok(arc)
}

/// Handler of Cuda context that was pushed as current.
/// On Drop will pop context from current.
pub struct ContextHandler<'a> {
    _ctx: &'a Context,
}

impl Drop for ContextHandler<'_> {
    fn drop(&mut self) {
        let _ = cuda_call!(cuCtxPopCurrent_v2(std::ptr::null_mut()));
    }
}

/// Cuda Context.
pub struct Context {
    context: cuda_driver_sys::CUcontext,
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    /// Create Context on device `dev`. It is recommended to use zeroed `flags`.
    pub fn new(dev: CuDevice, flags: u32) -> Result<Context, Error> {
        let mut ctx = Context {
            context: std::ptr::null_mut(),
        };

        cuda_call!(cuCtxCreate_v2(
            &mut ctx.context as *mut CUcontext,
            flags,
            dev.device
        ))
        .map(|_| ctx)
    }

    /// Get Cuda API version.
    pub fn get_api_version(&self) -> Result<u32, Error> {
        let mut ver = 0;
        cuda_call!(cuCtxGetApiVersion(self.context, &mut ver as *mut u32)).map(|_| ver)
    }

    /// Make this context current.
    pub fn make_current(&self) -> Result<ContextHandler<'_>, Error> {
        cuda_call!(cuCtxPushCurrent_v2(self.context))?;

        Ok(ContextHandler { _ctx: self })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.context.is_null() {
            let _ = cuda_call!(cuCtxDestroy_v2(self.context));
        }
    }
}

/// Cuda representation of the device.
#[derive(Debug, Clone, Copy, Default)]
pub struct CuDevice {
    pub device: CUdevice,
}

impl CuDevice {
    /// Create new device with id `ordinal`.
    pub fn new(ordinal: c_int) -> Result<CuDevice, Error> {
        let mut d = CuDevice { device: 0 };

        cuda_call!(cuDeviceGet(&mut d.device as *mut i32, ordinal)).map(|_| d)
    }

    /// Get attributes of the device.
    pub fn get_attribute(&self, attr: CUdevice_attribute) -> Result<c_int, Error> {
        let mut pi = 0;

        cuda_call!(cuDeviceGetAttribute(&mut pi as *mut i32, attr, self.device)).map(|_| pi)
    }

    /// Get name of the device.
    pub fn get_name(&self) -> Result<String, Error> {
        let mut name = vec![0; 256];

        cuda_call!(
            cuDeviceGetName(name.as_mut_ptr() as *mut _, 256, self.device,),
            from_char_array(name.as_mut_ptr())
        )
    }

    /// Get total mem of the device.
    pub fn get_total_mem(&self) -> Result<usize, Error> {
        let mut val = 0;

        cuda_call!(cuDeviceTotalMem_v2(
            &mut val as *mut usize as *mut _,
            self.device
        ))
        .map(|_| val)
    }
}
