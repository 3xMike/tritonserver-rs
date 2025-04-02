use std::ptr::null;

use crate::{sys, Error};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
/// Metric format types.
pub enum Format {
    /// Base points to a single multiline
    /// string that gives a text representation of the metrics in
    /// prometheus format.
    Prometheus = sys::tritonserver_metricformat_enum_TRITONSERVER_METRIC_PROMETHEUS,
}

/// Server metrics object.
pub struct Metrics(pub(crate) *mut sys::TRITONSERVER_Metrics);

impl Metrics {
    /// Get a buffer containing the metrics in the specified format.
    pub fn formatted(&self, format: Format) -> Result<&[u8], Error> {
        #[cfg(target_arch = "x86_64")]
        let mut ptr = null::<i8>();
        #[cfg(target_arch = "aarch64")]
        let mut ptr = null::<u8>();
        let mut size: usize = 0;

        triton_call!(sys::TRITONSERVER_MetricsFormatted(
            self.0,
            format as _,
            &mut ptr as *mut _,
            &mut size as *mut _,
        ))?;

        assert!(!ptr.is_null());
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const u8, size) })
    }
}

impl Drop for Metrics {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                sys::TRITONSERVER_MetricsDelete(self.0);
            }
        }
    }
}
