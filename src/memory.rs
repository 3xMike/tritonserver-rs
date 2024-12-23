//! Module responsible for memory allocation and assignments.
//!
//! **NOTE**: some functions that uses CUDA must be run in synchronous context +
//! [crate::context::Context] must me pushed as current. \
//! To simplify satisfaction of this requirement, those methods can be run with [crate::run_in_context] or [crate::run_in_context_sync] macro.
//! ```
//! run_in_context!(0, Buffer::alloc<f32>(10, MemoryType::Gpu))
//! ```

use core::slice;
use std::{
    ffi::CStr,
    fmt::Debug,
    intrinsics::copy_nonoverlapping,
    mem::{size_of_val, transmute},
    ops::{Bound, RangeBounds},
};

#[cfg(feature = "gpu")]
use cuda_driver_sys::{
    cuMemAllocHost_v2, cuMemAlloc_v2, cuMemFreeHost, cuMemFree_v2, cuMemcpyDtoD_v2,
    cuMemcpyDtoH_v2, cuMemcpyHtoD_v2, CUdeviceptr,
};
use libc::{c_void, calloc, free};

use crate::{
    error::{Error, ErrorCode, CSTR_CONVERT_ERROR_PLUG},
    sys, to_cstring,
};

macro_rules! impl_sample {
    ($type:ty, $data:expr) => {
        impl private::Sealed for $type {}

        impl Sample for $type {
            const DATA_TYPE: DataType = $data;
        }
    };
}

mod private {
    pub trait Sealed: Clone + Copy {}
}

/// Trait of objects that can be stored in Buffer.
pub trait Sample: private::Sealed {
    const DATA_TYPE: DataType;
}

/// Tensor data types recognized by TRITONSERVER.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DataType {
    Invalid = sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INVALID,
    Bool = sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BOOL,
    Uint8 = sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT8,
    Uint16 = sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT16,
    Uint32 = sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT32,
    Uint64 = sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT64,
    Int8 = sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT8,
    Int16 = sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT16,
    Int32 = sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT32,
    Int64 = sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT64,
    Fp16 = sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP16,
    Fp32 = sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32,
    Fp64 = sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP64,
    Bytes = sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BYTES,
    Bf16 = sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BF16,
}

#[derive(Clone, Copy)]
pub struct Byte(pub u8);

impl_sample!(bool, DataType::Bool);
impl_sample!(u8, DataType::Uint8);
impl_sample!(Byte, DataType::Bytes);
impl_sample!(u16, DataType::Uint16);
impl_sample!(u32, DataType::Uint32);
impl_sample!(u64, DataType::Uint64);

impl_sample!(i8, DataType::Int8);
impl_sample!(i16, DataType::Int16);
impl_sample!(i32, DataType::Int32);
impl_sample!(i64, DataType::Int64);

impl_sample!(half::f16, DataType::Fp16);
impl_sample!(half::bf16, DataType::Bf16);
impl_sample!(f32, DataType::Fp32);
impl_sample!(f64, DataType::Fp64);

impl DataType {
    /// Get the string representation of a data type.
    pub fn as_str(self) -> &'static str {
        let ptr = unsafe { sys::TRITONSERVER_DataTypeString(self as u32) };
        unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .unwrap_or(CSTR_CONVERT_ERROR_PLUG)
    }

    /// Get the size of a Triton datatype in bytes. Zero is returned for [DataType::Bytes] because it have variable size.
    pub fn size(self) -> u32 {
        unsafe { sys::TRITONSERVER_DataTypeByteSize(self as u32) }
    }
}

impl TryFrom<&str> for DataType {
    type Error = Error;
    /// Get the Triton datatype corresponding to a string representation of a datatype.
    fn try_from(name: &str) -> Result<Self, Self::Error> {
        let name = to_cstring(name)?;
        let data_type = unsafe { sys::TRITONSERVER_StringToDataType(name.as_ptr()) };
        if data_type != sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INVALID {
            Ok(unsafe { transmute::<u32, crate::memory::DataType>(data_type) })
        } else {
            Err(Error::new(ErrorCode::InvalidArg, ""))
        }
    }
}

/// Types of memory recognized by TRITONSERVER.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[repr(u32)]
pub enum MemoryType {
    Cpu = sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
    Pinned = sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU_PINNED,
    Gpu = sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_GPU,
}

impl MemoryType {
    /// Get the string representation of a memory type.
    pub fn as_str(self) -> &'static str {
        let ptr = unsafe { sys::TRITONSERVER_MemoryTypeString(self as u32) };
        unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .unwrap_or(CSTR_CONVERT_ERROR_PLUG)
    }
}

/// Representation of GPU based cuda array.
///
/// Does not delete array on drop.
#[cfg(feature = "gpu")]
pub struct CudaArray {
    pub ptr: CUdeviceptr,
    pub len: usize,
}

/// Data storring buffer.
///
/// Deletes data on drop.
// # Safety
// As long as no one changes the owned field
// and there is no overlapped buffers, they are thread safe. \
// Since no one can change the state of Buffer using reference
// and ptr pointing on heap/gpu, it's Sync.
#[derive(Debug)]
pub struct Buffer {
    pub(crate) ptr: *mut c_void,
    // Byte size,
    pub(crate) len: usize,
    pub(crate) data_type: DataType,
    pub(crate) memory_type: MemoryType,
    /// Should we execute the Drop or not.
    pub(crate) owned: bool,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

/// Buffer creation section.
impl Buffer {
    /// Try clone buffer content.
    ///
    /// **Note**: If memory type is not Cpu, should be called in sync with cuda context pinned (check module level documentation for more info).
    pub fn try_clone(&self) -> Result<Self, Error> {
        self.check_mem_type_feature()?;

        let sample_count = self.len / self.data_type.size() as usize;
        let mut res = Buffer::alloc_with_data_type(sample_count, self.memory_type, self.data_type)?;

        if self.memory_type == MemoryType::Gpu {
            #[cfg(feature = "gpu")]
            res.copy_from_cuda_array(0, unsafe { self.get_cuda_array() })?;
        } else {
            res.copy_from_slice(0, self.bytes())?;
        }

        Ok(res)
    }

    /// Allocate new buffer of requested memory type.\
    /// `count`: size of buffer in `T` units (i.e. 128 chunks of f32 (that has byte size 512) should be allocated with `count=128`).\
    /// `memory_type`: Cpu/Pinned/Gpu.
    ///
    /// **Note**: If memory type is not Cpu, should be called in sync with cuda context pinned (check module level documentation for more info).
    pub fn alloc<T: Sample>(count: usize, memory_type: MemoryType) -> Result<Self, Error> {
        Self::alloc_with_data_type(count, memory_type, T::DATA_TYPE)
    }

    /// Allocate new buffer of requested memory type.\
    /// `count`: size of buffer in `T` units (i.e. 128 chunks of f32 (that has byte size 512) should be allocated with `count=128`).\
    /// `memory_type`: Cpu/Pinned/Gpu.\
    /// `data_type`: type of Samples.
    ///
    /// **Note**: If memory type is not Cpu, should be called in sync with cuda context pinned (check module level documentation for more info).
    pub fn alloc_with_data_type(
        count: usize,
        memory_type: MemoryType,
        data_type: DataType,
    ) -> Result<Self, Error> {
        let data_type_size = data_type.size() as usize;
        let size = count * data_type_size;

        let ptr = match memory_type {
            MemoryType::Cpu => Ok::<_, Error>(unsafe { calloc(count as _, data_type_size) }),
            MemoryType::Pinned => {
                #[cfg(not(feature = "gpu"))]
                return Err(Error::wrong_type(memory_type));
                #[cfg(feature = "gpu")]
                {
                    let mut data = std::ptr::null_mut::<c_void>();
                    cuda_call!(cuMemAllocHost_v2(&mut data, size))?;
                    Ok(data)
                }
            }
            MemoryType::Gpu => {
                #[cfg(not(feature = "gpu"))]
                return Err(Error::wrong_type(memory_type));
                #[cfg(feature = "gpu")]
                {
                    let mut data = 0;
                    cuda_call!(cuMemAlloc_v2(&mut data, size))?;
                    Ok(data as *mut c_void)
                }
            }
        }?;

        if ptr.is_null() {
            Err(Error::new(
                ErrorCode::Internal,
                format!("OutOfMemory. {memory_type:?}"),
            ))
        } else {
            Ok(Buffer {
                ptr,
                len: size,
                data_type,
                memory_type,
                owned: true,
            })
        }
    }

    /// Create CPU buffer of data type `T::DARA_TYPE` from `slice` of T.
    pub fn from<T: Sample, S: AsRef<[T]>>(slice: S) -> Self {
        let slice = slice.as_ref();
        let ptr = unsafe {
            let ptr = calloc(slice.len(), std::mem::size_of::<T>()) as *mut T;
            copy_nonoverlapping(slice.as_ptr(), ptr, slice.len());
            ptr
        };

        Buffer {
            ptr: ptr as *mut _,
            len: size_of_val(slice),
            data_type: T::DATA_TYPE,
            memory_type: MemoryType::Cpu,
            owned: true,
        }
    }
}

/// Create GPU buffers of [DataType::Uint8] from [CudaArray].
/// Result memory type will be [MemoryType::Gpu].
///
/// Note that nothing is allocated on this call, meaning that result buffer will just point on
/// data provided by argument.
#[cfg(feature = "gpu")]
impl From<CudaArray> for Buffer {
    fn from(value: CudaArray) -> Self {
        Buffer {
            ptr: value.ptr as *mut c_void,
            len: value.len,
            data_type: DataType::Uint8,
            memory_type: MemoryType::Gpu,
            owned: true,
        }
    }
}

/// Create [CudaArray] from [Buffer].
///
/// Buffer destructor will not be invoked so data will be safe.
#[cfg(feature = "gpu")]
impl From<Buffer> for CudaArray {
    fn from(value: Buffer) -> CudaArray {
        let res = CudaArray {
            ptr: value.ptr as _,
            len: value.len,
        };
        std::mem::forget(value);
        res
    }
}

/// Buffer metadata section.
impl Buffer {
    /// Get memory type of storred data.
    pub fn memory_type(&self) -> MemoryType {
        self.memory_type
    }

    /// Get data type of storred data.
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Get byte size of data.
    pub fn size(&self) -> usize {
        self.len
    }

    /// True if not containing any data.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Buffer data permutation section.
impl Buffer {
    /// Copy `source` content to self from the `offset` position.\
    /// Returns error if offset + size_of_val(source) > self.size().
    ///
    /// `offset`: offset (in bytes) from the beginning of the Buffer to location to copy `source` to.
    /// `source`: slice of Samples.
    ///
    /// **Note**: If self.memory_type is not Cpu, should be called in sync with cuda context pinned (check module level documentation for more info).
    pub fn copy_from_slice<S: AsRef<[T]>, T: Sample>(
        &mut self,
        offset: usize,
        source: S,
    ) -> Result<(), Error> {
        self.check_mem_type_feature()?;

        let slice = source.as_ref();

        let byte_size = size_of_val(slice);

        if self.len < byte_size + offset {
            return Err(Error::new(
                ErrorCode::Internal,
                format!(
                    "copy_from_slice error: size mismatch! (required {}, buffer len {})",
                    byte_size + offset,
                    self.len
                ),
            ));
        }

        match self.memory_type {
            MemoryType::Cpu | MemoryType::Pinned => unsafe {
                copy_nonoverlapping(slice.as_ptr(), self.ptr.byte_add(offset) as _, slice.len());
            },
            MemoryType::Gpu => {
                #[cfg(feature = "gpu")]
                cuda_call!(cuMemcpyHtoD_v2(
                    self.ptr as CUdeviceptr + offset as CUdeviceptr,
                    slice.as_ptr() as _,
                    byte_size
                ))?;
            }
        }
        Ok(())
    }

    /// Copy `source` content to self from the `offset` position.\
    /// Returns error if offset + source.len > self.size().
    ///
    /// `offset`: offset (in bytes) from the beginning of the Buffer to location to copy `source` to.
    /// `source`: cuda array.
    ///
    /// **Note**: This method should be called in sync with cuda context pinned (check module level documentation for more info).
    #[cfg(feature = "gpu")]
    pub fn copy_from_cuda_array(&mut self, offset: usize, source: CudaArray) -> Result<(), Error> {
        let CudaArray { ptr, len } = source;

        if len + offset > self.len {
            return Err(Error::new(
                ErrorCode::Internal,
                format!(
                    "copy_from_cuda_array error: size mismatch (buffer len {}, required {})",
                    self.len,
                    len + offset
                ),
            ));
        }

        match self.memory_type {
            MemoryType::Pinned | MemoryType::Cpu => {
                cuda_call!(cuMemcpyDtoH_v2(
                    self.ptr.byte_add(offset),
                    ptr as CUdeviceptr,
                    len
                ))?;
            }
            MemoryType::Gpu => {
                cuda_call!(cuMemcpyDtoD_v2(
                    self.ptr as CUdeviceptr + offset as CUdeviceptr,
                    ptr as CUdeviceptr,
                    len
                ))?;
            }
        }
        Ok(())
    }

    /// Move this Buffer content to CPU memory.
    ///
    /// **Note**: If self.memory_type() is not Cpu, method should be called in sync with cuda context pinned (check module level documentation for more info).
    pub fn into_cpu(self) -> Result<Self, Error> {
        self.into_mem_type(MemoryType::Cpu)
    }

    /// Move this Buffer content to Pinned memory.
    ///
    /// **Note**: This method should be called in sync with cuda context pinned (check module level documentation for more info).
    #[cfg(feature = "gpu")]
    pub fn into_pinned(self) -> Result<Self, Error> {
        self.into_mem_type(MemoryType::Pinned)
    }

    /// Move this Buffer content to Gpu memory.
    ///
    /// **Note**: This method should be called in sync with cuda context pinned (check module level documentation for more info).
    #[cfg(feature = "gpu")]
    pub fn into_gpu(self) -> Result<Self, Error> {
        self.into_mem_type(MemoryType::Gpu)
    }

    fn into_mem_type(self, mem_type: MemoryType) -> Result<Self, Error> {
        self.check_mem_type_feature()?;

        if self.memory_type == mem_type {
            return Ok(self);
        }

        let sample_count = self.len / self.data_type.size() as usize;
        let mut res = Buffer::alloc_with_data_type(sample_count, mem_type, self.data_type)?;

        if self.memory_type == MemoryType::Gpu {
            #[cfg(feature = "gpu")]
            res.copy_from_cuda_array(0, unsafe { self.get_cuda_array() })?;
        } else {
            res.copy_from_slice(0, self.bytes())?;
        }
        Ok(res)
    }
}

/// Obtaining buffer content section.
impl Buffer {
    /// Get buffer content as bytes.
    ///
    /// Will return nothing if self.memory_type == Gpu. Use [Buffer::get_owned_slice] instead.
    pub fn bytes(&self) -> &[u8] {
        if self.memory_type == MemoryType::Gpu {
            log::warn!("Use bytes() on Gpu Buffer. empty slice will be returned");
            return &[];
        }

        unsafe { slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }

    /// Get buffer mutable content as bytes.
    ///
    /// Will return nothing if self.memory_type == Gpu. [Buffer::get_cuda_array] can be used instead to implement this logic.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        if self.memory_type == MemoryType::Gpu {
            log::warn!("Use bytes_mut() on Gpu Buffer. empty slice will be returned");
            return &mut [];
        }

        unsafe { slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }

    /// Get content of the buffer as host located bytes.\
    /// `range`: part of the buffer to return.
    pub fn get_owned_slice<Range: RangeBounds<usize> + Debug>(
        &self,
        range: Range,
    ) -> Result<Vec<u8>, Error> {
        self.check_mem_type_feature()?;

        let left = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(pos) => *pos,
            Bound::Excluded(pos) => *pos + 1,
        };
        let right = match range.end_bound() {
            Bound::Unbounded => self.len,
            Bound::Included(pos) => *pos + 1,
            Bound::Excluded(pos) => *pos,
        };

        if right > self.len {
            return Err(Error::new(
                ErrorCode::InvalidArg,
                format!(
                    "get_slice invalid range: {range:?}, buffer len is: {}",
                    self.len
                ),
            ));
        }

        if self.memory_type != MemoryType::Gpu {
            Ok(self.bytes()[left..right].to_vec())
        } else {
            let mut res = Vec::with_capacity(right - left);
            #[cfg(feature = "gpu")]
            cuda_call!(cuMemcpyDtoH_v2(
                res.as_mut_ptr() as _,
                self.ptr as CUdeviceptr + left as CUdeviceptr,
                right - left
            ))?;

            unsafe { res.set_len(self.len) };
            Ok(res)
        }
    }

    /// Get content of the GPU based buffer.
    /// # Panics
    /// Panics if self.memory_type != Gpu.
    /// # Safety
    /// Returned struct points to the same location as buffer, so the rules are the same as sharing *mut on an object. \
    /// Be careful: Buffer will delete data on drop, so be afraid of double memory free.
    /// Also any shinenigans with the data during the inference are forbidden: Triton must have exclusive write ascess to data during the inference.
    #[cfg(feature = "gpu")]
    pub unsafe fn get_cuda_array(&self) -> CudaArray {
        if self.memory_type != MemoryType::Gpu {
            panic!("Invoking get_cuda_array for non GPU-based buffer");
        }

        CudaArray {
            ptr: self.ptr as _,
            len: self.len,
        }
    }

    fn check_mem_type_feature(&self) -> Result<(), Error> {
        #[cfg(not(feature = "gpu"))]
        if self.memory_type != MemoryType::Cpu {
            return Err(Error::wrong_type(self.memory_type));
        }
        Ok(())
    }
}

impl<T: Sample> AsRef<[T]> for Buffer {
    /// Converts this type into a shared reference on the slice of T.
    ///
    /// Will return nothing if self.memory_type == Gpu.
    /// # Panics
    /// Panics if T does not match Buffer data type.
    fn as_ref(&self) -> &[T] {
        if T::DATA_TYPE != self.data_type {
            panic!(
                "Buffer data_type {:?} != target slice data_type: {:?}",
                self.data_type,
                T::DATA_TYPE
            )
        }

        if self.memory_type == MemoryType::Gpu {
            log::warn!("Use as_ref() on Gpu Buffer. empty slice will be returned");
            return &[];
        }

        unsafe { slice::from_raw_parts(self.ptr as *const T, self.len) }
    }
}

impl<T: Sample> AsMut<[T]> for Buffer {
    /// Converts this type into a mutable reference on the slice of T.
    ///
    /// Will return nothing if self.memory_type == Gpu.
    /// # Panics
    /// Panics if T does not match Buffer data type.
    fn as_mut(&mut self) -> &mut [T] {
        if T::DATA_TYPE != self.data_type {
            panic!(
                "Buffer data_type {:?} != target slice data_type: {:?}",
                self.data_type,
                T::DATA_TYPE
            )
        }

        if self.memory_type == MemoryType::Gpu {
            log::warn!("Use as_mut() on Gpu Buffer. empty slice will be returned");
            return &mut [];
        }

        unsafe { slice::from_raw_parts_mut(self.ptr as *mut T, self.len) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe {
                match self.memory_type {
                    MemoryType::Cpu => {
                        free(self.ptr);
                    }
                    MemoryType::Pinned => {
                        #[cfg(feature = "gpu")]
                        cuMemFreeHost(self.ptr);
                    }
                    MemoryType::Gpu => {
                        #[cfg(feature = "gpu")]
                        cuMemFree_v2(self.ptr as CUdeviceptr);
                    }
                }
            }
        }
    }
}
