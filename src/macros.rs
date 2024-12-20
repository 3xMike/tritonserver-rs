#[cfg(feature = "gpu")]
/// Run cuda method and get the Result<(), tritonserver_rs::Error> instead of cuda_driver_sys::CUresult.
macro_rules! cuda_call {
    ($expr: expr) => {{
        #[allow(clippy::macro_metavars_in_unsafe)]
        let res = unsafe { $expr };

        if res != cuda_driver_sys::CUresult::CUDA_SUCCESS {
            Err($crate::error::Error::new(
                $crate::error::ErrorCode::Internal,
                format!("Cuda result: {:?}", res),
            ))
        } else {
            std::result::Result::<_, $crate::error::Error>::Ok(())
        }
    }};
    ($expr: expr, $val: expr) => {{
        #[allow(clippy::macro_metavars_in_unsafe)]
        let res = unsafe { $expr };

        if res != cuda_driver_sys::CUresult::CUDA_SUCCESS {
            Err($crate::error::Error::new(
                $crate::error::ErrorCode::Internal,
                format!("Cuda result: {:?}", res),
            ))
        } else {
            std::result::Result::<_, $crate::error::Error>::Ok($val)
        }
    }};
}

/// Run triton method and get the Result<(), tritonserver_rs::Error> instead of cuda_driver_sys::CUresult.
macro_rules! triton_call {
    ($expr: expr) => {{
        #[allow(clippy::macro_metavars_in_unsafe)]
        let res = unsafe { $expr };

        if res.is_null() {
            std::result::Result::<(), $crate::error::Error>::Ok(())
        } else {
            std::result::Result::<(), $crate::error::Error>::Err(res.into())
        }
    }};
    ($expr: expr, $val: expr) => {{
        #[allow(clippy::macro_metavars_in_unsafe)]
        let res = unsafe { $expr };

        if res.is_null() {
            std::result::Result::<_, $crate::error::Error>::Ok($val)
        } else {
            std::result::Result::<_, $crate::error::Error>::Err(res.into())
        }
    }};
}

/// Run cuda code (which should be run in sync + cuda context pinned) in asynchronous context.
///
/// First argument is an id of device to run function on; second is the code to run. All the variables will be moved.
///
/// If "gpu" feature is off just runs a code without context/blocking.
#[macro_export]
macro_rules! run_in_context {
    ($val: expr, $expr: expr) => {{
        #[cfg(feature = "gpu")]
        {
            tokio::task::spawn_blocking(move || {
                let ctx = $crate::get_context($val)?;
                let _handle = ctx.make_current()?;
                $expr
            })
            .await
            .expect("tokio failed to join thread")
        }
        #[cfg(not(feature = "gpu"))]
        $expr
    }};
}

/// Run cuda code (which should be run in sync + cuda context pinned).
///
/// First argument is an id of device to run function on; second is the code to run.
///
/// If "gpu" feature is off just runs a code without context/blocking.
#[macro_export]
macro_rules! run_in_context_sync {
    ($val: expr, $expr: expr) => {{
        #[cfg(feature = "gpu")]
        {
            let ctx = $crate::get_context($val)?;
            let _handle = ctx.make_current()?;
            $expr
        }
        #[cfg(not(feature = "gpu"))]
        $expr
    }};
}
