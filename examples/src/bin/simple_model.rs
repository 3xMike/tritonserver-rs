use std::time::Duration;

use anyhow::Result;
use config_manager::{config, ConfigInit};
use tritonserver_rs::{
    memory::DataType,
    trace::{Activity, TensorTraceHandler, Trace, TraceHandler},
    Allocator, Buffer, Error, MemoryType, Request, Response,
};

use triton_examples::*;

#[tokio::main]
async fn main() -> Result<()> {
    let pipe = SimplePipe::parse()?;
    println!("{pipe:?}");

    run_pipeline(pipe).await
}

#[derive(Debug)]
#[config]
struct SimplePipe {
    #[source(clap(long, short), default = "DEFAULT_BACKEND_PATH.into()")]
    backend: String,
    #[source(clap(long, short), default = "\"models/simple_model\".into()")]
    model_path: String,
    #[source(clap(long, short), default = "vec![1.0,2.0,3.0,4.0,5.0,6.0]")]
    input_vector: Vec<f32>,
}

impl Pipeline for SimplePipe {
    fn backends_path(&self) -> String {
        self.backend.clone()
    }

    fn model_repo(&self) -> String {
        self.model_path.clone()
    }

    fn model_name(&self) -> String {
        "adder".into()
    }

    fn add_inputs(&mut self, request: &mut Request) -> Result<()> {
        request.add_trace(Trace::new_with_handle(0, Some(TraceH), Some(TraceH))?);
        request.add_allocator(Box::new(CustomAllocator));

        request.add_input_with_dims(
            "input",
            Buffer::from(&self.input_vector),
            [1, 1, 1, self.input_vector.len() as _],
        )?;
        Ok(())
    }

    fn parse_result(&mut self, result: Response) -> Result<()> {
        let out = result.get_output("output").unwrap();
        let result: f32 = out.get_buffer().as_ref()[0];
        log::info!("Sum by dims of {:?} is: {result}", self.input_vector);
        Ok(())
    }
}

struct TraceH;

impl TraceHandler for TraceH {
    fn trace_activity(
        &self,
        trace: &tritonserver_rs::trace::Trace,
        event: Activity,
        event_time: Duration,
    ) {
        log::info!(
            "Tracing activities: Trace_id: {}, event: {event:?}, event_time_secs: {}",
            trace.id().unwrap(),
            event_time.as_nanos()
        );
        if event == Activity::ComputeStart {
            let child_trace = trace.spawn_child().unwrap();

            let id = child_trace.id().unwrap();
            log::info!("Computations start, spawning new Trace with id: {id}");
            std::thread::spawn(move || {
                child_trace
                    .report_activity(event_time, "Child trace born")
                    .unwrap();
                std::thread::sleep(Duration::from_secs(1));
                child_trace
                    .report_activity(event_time + Duration::from_secs(1), "Child trace died")
                    .unwrap();
            });
        }
    }
}

impl TensorTraceHandler for TraceH {
    fn trace_tensor_activity(
        &self,
        trace: &Trace,
        event: Activity,
        _tensor_data: &tritonserver_rs::Buffer,
        tensor_shape: tritonserver_rs::message::Shape,
    ) {
        log::info!(
            "Tracing Tensor Activity: Trace_id: {}, event: {event:?}, tensor name: {}",
            trace.id().unwrap(),
            tensor_shape.name
        );
    }
}

/// Allocator that allocate only CPU memory, maximum of 10Mb
struct CustomAllocator;

#[async_trait::async_trait]
impl Allocator for CustomAllocator {
    async fn allocate(
        &mut self,
        tensor_name: String,
        requested_memory_type: MemoryType,
        byte_size: usize,
        data_type: DataType,
    ) -> Result<Buffer, tritonserver_rs::Error> {
        log::info!("Triton requested {byte_size} bytes of {requested_memory_type:?} of {data_type:?} for output {tensor_name}. But we have only CPU, hope Triton will parse it");
        if requested_memory_type == MemoryType::Gpu {
            return Err(Error::new(
                tritonserver_rs::ErrorCode::Unavailable,
                "Can't allocate GPU for this model",
            ));
        }

        if byte_size > 10 * (1 << 20) {
            return Err(Error::new(
                tritonserver_rs::ErrorCode::Unavailable,
                "Can't allocate more than 10 Mb",
            ));
        }

        Buffer::alloc_with_data_type(
            byte_size / data_type.size() as usize,
            MemoryType::Cpu,
            data_type,
        )
    }
}
