# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1](https://github.com/3xMike/tritonserver-rs/tags/0.2.1) - 2024-12-24
### Fixed:
- Error on context::get_context()

## [0.2.0](https://github.com/3xMike/tritonserver-rs/tags/0.2.0) - 2024-12-23
### Changed:
- Triton C-API version to 1.33.
- Minimal TritonInferenceServer container version to 24.07.
- Folowwing bitflag structs with consts now enums (variant names changed to UpperCamelCase):
    - metrics::Format,
    - options::{Control, Limit},
    - parameter::TritonParameterType,
    - request::Sequence,
    - server::{Batch, Transaction}.
- Allocator trait now has 2 optional methods that implements queriing logic: `enable_queries()` and `pre_allocation_query()`.
- request:
    - Allocator::allocate now takes 4th argument: DataType,
    - DefaultAllocator now has no methods and impl Copy,
    - Request::get_id() now returns `Result<String>` instead of `Result<&str>`,
    - Request::get_correlation_id_as_str() now is called get_correlation_id_as_string() and returns `Result<String>` instead of `Result<&str>`.
- response::Response::id() now returns `Result<String>` instead of `Result<&str>`.
- trace:
    - enum Level deleted,
    - Trace::new_with_handle() now takes parent_id, `Option<TraceHandler>` and `Option<TensorTraceHandler>` ,
    - trait TraceHandler now requires Send + Sync
    - Trace::{get_id(), get_parent_id(), get_model_name(), get_model_version()} now is called id(), parent_id(), model_name() and model_version() respectably,
    - Trace::get_model_name() now returns `Result<String>` instead of `Result<&str>`.

### Added:
- enum variants memory::DataType::{Bf16, Invalid}.
- memory::Buffer::alloc_with_data_type().
- options:
    - enum Format,
    - enum InstanceGroupKind,
    - methods Options::{ \
        &emsp;model_config_name(), cuda_virtual_address_size(), response_cache_directory(), \
        &emsp;response_cache_config(), model_load_thread_count(), model_retry_count(), \
        &emsp;peer_access(), model_namespacing(), log_file(), log_format(), \
        &emsp;cpu_metrics(), model_load_device_limit(), metrics_config() \
    }.
- enum variant parameter::TritonParameterType::Double.
- request:
    - methods Request::{remove_input(), remove_all_inputs(), set_parameter()},
    - Request now can be cancelled by dropping ResponseFuture.
- server::Server::{set_exit_timeout(), register_model_repo(), unregister_model_repo(), is_log_enabled()}.
- trace:
    - better documentation,
    - enum variants Activity::{TensorQueueInput, TensorBackendInput, TensorBackendOutput, CustomActivity},
    - impl TraceHandler for (),
    - trait TensorTraceHandler,
    - impl PartialEq for Trace,
    - const NOOP, 
    - methods Trace::{report_activity(), request_id(), spawn_child(), set_context(), context()}.

### Depricated:
- options::Options::response_cache_byte_size().
- trace::Level::{MIN, MAX}.

### Notes:
Following Tritonserver C-API structures and their functions was not implemented, because the author saw no benefit of them: BufferAttributes. It might be implement by the first Issue/Pull Request that will give clear explainations why it might be needed.

Metric (not to be confused with Metrics) will not be implemented because there are better solutions on Rust.

## [0.1.0](https://github.com/3xMike/tritonserver-rs/tags/0.1.0) - 2024-12-04
INITIAL STATE
