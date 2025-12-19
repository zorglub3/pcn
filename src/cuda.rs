use cudarc::driver::CudaContext;
use thiserror::Error;

const CU_SRC: &str = include_str!("cuda/kernel.cu");

#[derive(Error, Debug)]
pub enum CudaError {
    #[error("Cuda driver error: {0}")]
    CudaDriverError(#[from] DriverError),
}

pub struct CudaInterface {
    context: CudaContext,
}

impl CudaInterface {
    pub fn new() -> Result<CudaInterface, CudaError> {
        let context = CudaContext::new(0)?;
        Self { context }
    }
}
