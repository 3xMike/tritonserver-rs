use std::{
    collections::HashMap,
    ffi::c_void,
    os::raw::{c_char, c_uint},
    ptr::null_mut,
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc,
    },
};

use log::{error, trace};
use tokio::{runtime::Handle, sync::RwLock};

use crate::{
    error::{Error, ErrorCode},
    from_char_array,
    memory::{Buffer, DataType, MemoryType},
    request::Allocator as AllocTrait,
    sys,
};

type Outputs = HashMap<String, Buffer>;

pub(crate) struct Inner {
    alloc: *mut sys::TRITONSERVER_ResponseAllocator,
    pub(crate) output_buffers: RwLock<Outputs>,
    alloc_called: AtomicBool,
    pub(crate) returned_buffers: AtomicU32,
    /// User is responsible for buffers allocation.
    custom_allocator: RwLock<Box<dyn AllocTrait>>,
    datatype_hints: HashMap<String, DataType>,
    /// To run async code in sync C fn
    runtime: Handle,
}

/// Response allocator object.
/// New allocator should be used for each request.
pub(crate) struct Allocator(pub(crate) Arc<Inner>);

impl Clone for Allocator {
    fn clone(&self) -> Self {
        Allocator(self.0.clone())
    }
}

impl Allocator {
    pub(crate) fn new(
        custom_allocator: Box<dyn AllocTrait>,
        datatype_hints: HashMap<String, DataType>,
        runtime: Handle,
    ) -> Result<Self, Error> {
        let mut ptr = null_mut::<sys::TRITONSERVER_ResponseAllocator>();
        triton_call!(sys::TRITONSERVER_ResponseAllocatorNew(
            &mut ptr as *mut _,
            Some(alloc),
            Some(release),
            None,
        ))?;

        assert!(!ptr.is_null());

        if custom_allocator.enable_queries() {
            triton_call!(sys::TRITONSERVER_ResponseAllocatorSetQueryFunction(
                ptr,
                Some(query)
            ))?;
        }

        Ok(Self(Arc::new(Inner {
            alloc: ptr,
            output_buffers: RwLock::new(HashMap::new()),
            alloc_called: AtomicBool::new(false),
            datatype_hints,
            returned_buffers: AtomicU32::new(0),
            custom_allocator: RwLock::new(custom_allocator),
            runtime,
        })))
    }

    pub(crate) fn is_alloc_called(&self) -> bool {
        self.0.alloc_called.load(Ordering::Relaxed)
    }

    pub(crate) fn get_allocator(&self) -> *mut sys::TRITONSERVER_ResponseAllocator {
        self.0.alloc
    }
}

unsafe impl Send for Allocator {}
/// # SAFETY
/// Inner is Send. It also Sync as far as no one
/// use get_allocator and modify ptr in different threads.
/// Since it's private, for user it's Send + Sync
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        if !self.alloc.is_null() {
            unsafe {
                sys::TRITONSERVER_ResponseAllocatorDelete(self.alloc);
            }
        }
    }
}

/// C-code calls this fn when it need a buffer,
/// to the embeddings in. It will take the ownership until the release().
unsafe extern "C" fn alloc(
    _allocator: *mut sys::TRITONSERVER_ResponseAllocator,
    tensor_name: *const c_char,
    byte_size: libc::size_t,
    memory_type: sys::TRITONSERVER_MemoryType,
    // mem type id typically is the same as device id.
    memory_type_id: i64,
    userp: *mut c_void,
    buffer: *mut *mut c_void,
    buffer_userp: *mut *mut c_void,
    actual_memory_type: *mut sys::TRITONSERVER_MemoryType,
    actual_memory_type_id: *mut i64,
) -> *mut sys::TRITONSERVER_Error {
    let output_name = from_char_array(tensor_name);

    let mem_type = std::mem::transmute::<u32, MemoryType>(memory_type);

    log::trace!("Triton requested {byte_size} bytes of {mem_type:?} for output {output_name}",);

    let allocator = match unsafe { (userp as *const Allocator).as_ref() } {
        None => return Error::new(ErrorCode::Internal, "Got null userp in alloc method").ptr,
        Some(alloc) => alloc.clone(),
    };

    let data_type = match allocator.0.datatype_hints.get(&output_name) {
        Some(dt) => *dt,
        None => {
            return Error::new(
                ErrorCode::Internal,
                format!("Can't find datatype of tensor: {output_name}"),
            )
            .ptr
        }
    };

    // Оповещаем, что произошла алокация.
    allocator.0.alloc_called.store(true, Ordering::Relaxed);
    // Достаем буфер-пару, соответствующий указанному имени.

    let allocator_cloned = allocator.clone();
    let runtime = allocator.0.runtime.clone();
    let output = output_name.clone();
    let allocation_result = std::thread::spawn(move || {
        runtime.block_on(async move {
            allocator_cloned
                .0
                .custom_allocator
                .write()
                .await
                .allocate(output, mem_type, byte_size, data_type)
                .await
        })
    })
    .join()
    .unwrap();

    let users_buffer = match allocation_result {
        Ok(buf) => buf,
        Err(err) => {
            error!("Error in alloc method: {err}");
            return err.ptr;
        }
    };

    // Проверки, что буфер подходящий
    assert!(
        users_buffer.len >= byte_size,
        "User allocate smaller buffer ({}b), than required {byte_size} for output {output_name}.",
        users_buffer.len
    );
    let mem_type = std::mem::transmute::<u32, MemoryType>(memory_type);

    match (mem_type, users_buffer.memory_type) {
        (MemoryType::Cpu, MemoryType::Gpu) => panic!("Triton requested to alloc CPU memory for output {output_name} while user provided GPU buffer"),
        (MemoryType::Gpu, MemoryType::Cpu) => panic!("Triton requested to alloc GPU memory for output {output_name} while user provided CPU buffer"),
        _ => ()
    }

    let act_mem_type = users_buffer.memory_type;

    *actual_memory_type = act_mem_type as c_uint;
    *actual_memory_type_id = memory_type_id;
    *buffer = users_buffer.ptr;

    *buffer_userp = Box::into_raw(Box::new(ReleaseItems {
        allocator,
        allocated_buffer: users_buffer,
        allocated_tensor_name: output_name.to_string(),
    })) as *mut c_void;

    null_mut()
}

/// Items that flow from alloc fn to release fn.
struct ReleaseItems {
    allocator: Allocator,
    allocated_buffer: Buffer,
    allocated_tensor_name: String,
}

/// C-code calls release to give the ownership on output buffer back. \\
/// TIP: It calls it right after the Response delete method ([sys::TRITONSERVER_InferenceResponseDelete]).
unsafe extern "C" fn release(
    _allocator: *mut sys::TRITONSERVER_ResponseAllocator,
    buffer: *mut c_void,
    buffer_userp: *mut c_void,
    byte_size: libc::size_t,
    _memory_type: sys::TRITONSERVER_MemoryType,
    _memory_type_id: i64,
) -> *mut sys::TRITONSERVER_Error {
    trace!("release is called");
    assert!(!buffer_userp.is_null());

    let ReleaseItems {
        allocator,
        allocated_buffer,
        allocated_tensor_name,
    } = *Box::from_raw(buffer_userp as *mut ReleaseItems);

    assert!(byte_size <= allocated_buffer.len);
    assert_eq!(buffer, allocated_buffer.ptr);

    let alloc = allocator.clone();
    let runtime = allocator.0.runtime.clone();
    // Вставляем обратно использованный буфер
    std::thread::spawn(move || {
        runtime.block_on(async move {
            alloc
                .0
                .output_buffers
                .write()
                .await
                .insert(allocated_tensor_name, allocated_buffer);
        })
    })
    .join()
    .unwrap();

    allocator.0.returned_buffers.fetch_add(1, Ordering::Relaxed);
    trace!("release is ended");

    null_mut()
}

unsafe extern "C" fn query(
    _allocator: *mut sys::TRITONSERVER_ResponseAllocator,
    userp: *mut ::std::os::raw::c_void,
    tensor_name: *const ::std::os::raw::c_char,
    byte_size: *mut usize,
    memory_type: *mut sys::TRITONSERVER_MemoryType,
    _memory_type_id: *mut i64,
) -> *mut sys::TRITONSERVER_Error {
    let output_name = (!tensor_name.is_null()).then(|| from_char_array(tensor_name));
    let byte_size = (!byte_size.is_null()).then(|| *byte_size);
    let mem_type = std::mem::transmute::<u32, MemoryType>(*memory_type);

    let allocator = match unsafe { (userp as *const Allocator).as_ref() } {
        None => return Error::new(ErrorCode::Internal, "Got null userp in query method").ptr,
        Some(alloc) => alloc.clone(),
    };

    let allocator_cloned = allocator.clone();
    let runtime = allocator.0.runtime.clone();
    let allocation_result = std::thread::spawn(move || {
        runtime.block_on(async move {
            allocator_cloned
                .0
                .custom_allocator
                .write()
                .await
                .pre_allocation_query(output_name, byte_size, mem_type)
                .await
        })
    })
    .join()
    .unwrap();

    *memory_type = allocation_result as _;

    null_mut()
}
