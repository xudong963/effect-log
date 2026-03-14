pub mod error;
pub mod executor;
pub mod recovery;
pub mod registry;
pub mod storage;
pub mod types;
pub mod wal;

// Re-export primary API surface
pub use error::{EffectLogError, Result};
pub use executor::{EffectLog, EffectLogBuilder};
pub use recovery::recovery_strategy;
pub use registry::{RegisteredTool, ToolBuilder, ToolRegistry};
pub use storage::EffectStore;
pub use types::*;

pub use storage::memory::InMemoryStore;
#[cfg(feature = "sqlite")]
pub use storage::sqlite::SqliteStore;
