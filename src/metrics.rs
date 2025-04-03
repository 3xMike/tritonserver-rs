use std::{
    mem::transmute,
    ptr::{null, null_mut},
    sync::Arc,
};

use crate::{parameter::Parameter, sys, to_cstring, Error};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
/// Metric format types.
enum Format {
    /// Base points to a single multiline
    /// string that gives a text representation of the metrics in
    /// prometheus format.
    Prometheus = sys::tritonserver_metricformat_enum_TRITONSERVER_METRIC_PROMETHEUS,
}

/// Prometheus metrics object.
#[derive(Debug, Clone)]
pub struct PrometheusMetrics(pub(crate) Arc<*mut sys::TRITONSERVER_Metrics>);

unsafe impl Send for PrometheusMetrics {}
unsafe impl Sync for PrometheusMetrics {}

impl PrometheusMetrics {
    /// Get a buffer containing the metrics in the specified format.
    pub fn formatted(&self) -> Result<&[u8], Error> {
        let format = Format::Prometheus;

        #[cfg(target_arch = "x86_64")]
        let mut ptr = null::<i8>();
        #[cfg(target_arch = "aarch64")]
        let mut ptr = null::<u8>();
        let mut size: usize = 0;

        triton_call!(sys::TRITONSERVER_MetricsFormatted(
            *self.0,
            format as _,
            &mut ptr as *mut _,
            &mut size as *mut _,
        ))?;

        assert!(!ptr.is_null());
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const u8, size) })
    }
}

impl Drop for PrometheusMetrics {
    fn drop(&mut self) {
        if !self.0.is_null() && Arc::strong_count(&self.0) == 1 {
            unsafe { sys::TRITONSERVER_MetricsDelete(*self.0) };
        }
    }
}

/// Types of metrics recognized by TRITONSERVER.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum MetricKind {
    Counter = sys::TRITONSERVER_metrickind_enum_TRITONSERVER_METRIC_KIND_COUNTER,
    Gauge = sys::TRITONSERVER_metrickind_enum_TRITONSERVER_METRIC_KIND_GAUGE,
    Histogram = sys::TRITONSERVER_metrickind_enum_TRITONSERVER_METRIC_KIND_HISTOGRAM,
}

/// Family of the metrics.
///
/// Author note: the current state of [Metric], [MetricKind], [MetricFamily] is not kinda useful.
/// Added due to the politic to add every item of [original API](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).
/// It's worth considering using dedicated crates like [metrics](https://crates.io/crates/metrics).
#[derive(Debug, Clone)]
pub struct MetricFamily(Arc<*mut sys::TRITONSERVER_MetricFamily>);

unsafe impl Send for MetricFamily {}
unsafe impl Sync for MetricFamily {}

impl MetricFamily {
    /// Create a new metric family object.
    /// - `kind`: The type of metric family to create.
    /// - `name`: The name of the metric family seen when calling the metrics
    ///     endpoint.
    /// - `description`: The description of the metric family seen when
    ///     calling the metrics endpoint.
    pub fn new<N: AsRef<str>, D: AsRef<str>>(
        kind: MetricKind,
        name: N,
        description: D,
    ) -> Result<Self, Error> {
        let c_name = to_cstring(name)?;
        let descr = to_cstring(description)?;
        let mut res = null_mut::<sys::TRITONSERVER_MetricFamily>();
        triton_call!(
            sys::TRITONSERVER_MetricFamilyNew(
                &mut res as *mut _,
                kind as _,
                c_name.as_ptr(),
                descr.as_ptr()
            ),
            Self(Arc::new(res))
        )
    }

    /// Get the kind of the metric family.
    pub fn kind(&self) -> Result<MetricKind, Error> {
        let mut res = 0;
        triton_call!(sys::TRITONSERVER_GetMetricFamilyKind(*self.0, &mut res))?;
        Ok(unsafe { transmute::<u32, MetricKind>(res) })
    }
}

impl Drop for MetricFamily {
    fn drop(&mut self) {
        if !self.0.is_null() && Arc::strong_count(&self.0) == 1 {
            unsafe { sys::TRITONSERVER_MetricFamilyDelete(*self.0) };
        }
    }
}

/// Arguments to pass on creating [Metric]. Currently only [Self::set_histogram] is supported.
///
/// Author note: the current state of [Metric], [MetricKind], [MetricFamily] is not kinda useful.
/// Added due to the politic to add every item of [original API](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).
/// It's worth considering using dedicated crates like [metrics](https://crates.io/crates/metrics).
#[derive(Debug, Clone)]
pub struct MetricArgs(Arc<*mut sys::TRITONSERVER_MetricArgs>);

unsafe impl Send for MetricArgs {}
unsafe impl Sync for MetricArgs {}

impl MetricArgs {
    /// Create a new metric args object.
    pub fn new() -> Result<Self, Error> {
        let mut res = null_mut::<sys::TRITONSERVER_MetricArgs>();
        triton_call!(
            sys::TRITONSERVER_MetricArgsNew(&mut res),
            Self(Arc::new(res))
        )
    }

    /// Set metric args with histogram metric parameter.
    /// - `buckets`: The array of bucket boundaries for the expected range of
    ///     observed values.
    pub fn set_histogram<B: AsRef<[f64]>>(&mut self, buckets: B) -> Result<&mut Self, Error> {
        let buckets = buckets.as_ref();
        triton_call!(
            sys::TRITONSERVER_MetricArgsSetHistogram(*self.0, buckets.as_ptr(), buckets.len() as _),
            self
        )
    }
}

impl Drop for MetricArgs {
    fn drop(&mut self) {
        if !self.0.is_null() && Arc::strong_count(&self.0) == 1 {
            unsafe { sys::TRITONSERVER_MetricArgsDelete(*self.0) };
        }
    }
}

#[allow(dead_code)]
/// Metric: Counter, Gauge or Histogram.
///
/// Author note: the current state of [Metric], [MetricKind], [MetricFamily] is not kinda useful.
/// Added due to the politic to add every item of [original API](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).
/// It's worth considering using dedicated crates like [metrics](https://crates.io/crates/metrics).
#[derive(Debug, Clone)]
pub struct Metric(Arc<*mut sys::TRITONSERVER_Metric>, MetricFamily);

unsafe impl Send for Metric {}
unsafe impl Sync for Metric {}

impl Metric {
    /// Create a new metric object.
    /// - `family`: The metric family to add this new metric to.
    /// - `labels`: The array of labels to associate with this new metric.
    pub fn new<P: AsRef<[Parameter]>>(family: &MetricFamily, labels: P) -> Result<Self, Error> {
        let mut res = null_mut::<sys::TRITONSERVER_Metric>();
        let mut labels = labels
            .as_ref()
            .iter()
            .map(|p| *p.ptr as *const _)
            .collect::<Vec<_>>();
        triton_call!(
            sys::TRITONSERVER_MetricNew(
                &mut res,
                *family.0,
                labels.as_mut_ptr(),
                labels.len() as _
            ),
            Self(Arc::new(res), family.clone())
        )
    }

    /// Create a new metric object.
    /// - `family`: The metric family to add this new metric to.
    /// - `labels`: The array of labels to associate with this new metric.
    /// - `args`: Metric args that store additional arguments to construct
    ///     particular metric types, e.g. histogram.
    pub fn new_with_args<P: AsRef<[Parameter]>>(
        family: &MetricFamily,
        labels: P,
        args: &MetricArgs,
    ) -> Result<Self, Error> {
        let mut res = null_mut::<sys::TRITONSERVER_Metric>();
        let mut labels = labels
            .as_ref()
            .iter()
            .map(|p| *p.ptr as *const _)
            .collect::<Vec<_>>();
        triton_call!(
            sys::TRITONSERVER_MetricNewWithArgs(
                &mut res,
                *family.0,
                labels.as_mut_ptr(),
                labels.len() as _,
                *args.0 as *const _
            ),
            Self(Arc::new(res), family.clone())
        )
    }

    /// Get the kind of metric of its corresponding family.
    pub fn kind(&self) -> Result<MetricKind, Error> {
        let mut res = 0;
        triton_call!(sys::TRITONSERVER_GetMetricKind(*self.0, &mut res))?;
        Ok(unsafe { transmute::<u32, MetricKind>(res) })
    }

    /// Get the current value of a metric object.
    /// Supports metrics of kind [MetricKind::Counter]
    /// and [MetricKind::Gauge], and returns [Error::Unsupported](crate::ErrorCode::Unsupported)
    /// for other kinds.
    pub fn value(&self) -> Result<f64, Error> {
        let mut res = 0.;
        triton_call!(sys::TRITONSERVER_MetricValue(*self.0, &mut res), res)
    }

    /// Increment the current value of metric by value.
    /// Supports metrics of kind [MetricKind::Gauge] for any value,
    /// and [MetricKind::Counter] for non-negative values. Returns
    /// [Error::Unsupported](crate::ErrorCode::Unsupported) for other kinds
    /// and [Error::InvalidArg](crate::ErrorCode::InvalidArg) for negative values on a
    /// [MetricKind::Counter] metric.
    /// - `value`: The amount to increment the metric's value by.
    pub fn increment_by(&self, value: f64) -> Result<(), Error> {
        triton_call!(sys::TRITONSERVER_MetricIncrement(*self.0, value))
    }

    /// Set the current value of metric to value.
    /// Supports metrics of kind [MetricKind::Gauge] and returns
    /// [Error::Unsupported](crate::ErrorCode::Unsupported) for other kinds.
    ///
    /// - `value`: The amount to set metric's value to.
    pub fn set(&self, value: f64) -> Result<(), Error> {
        triton_call!(sys::TRITONSERVER_MetricSet(*self.0, value))
    }

    /// Sample an observation and count it to the appropriate bucket of a metric.
    /// Supports metrics of kind [MetricKind::Histogram] and returns
    /// [Error::Unsupported](crate::ErrorCode::Unsupported) for other kinds.
    /// - `value`: The amount for metric to sample observation.
    pub fn observe(&self, value: f64) -> Result<(), Error> {
        triton_call!(sys::TRITONSERVER_MetricObserve(*self.0, value))
    }
}

impl Drop for Metric {
    fn drop(&mut self) {
        if !self.0.is_null() && Arc::strong_count(&self.0) == 1 {
            unsafe { sys::TRITONSERVER_MetricDelete(*self.0) };
        }
    }
}
