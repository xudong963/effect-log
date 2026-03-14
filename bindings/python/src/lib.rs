use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use effect_log_core::storage::EffectStore;
use effect_log_core::{EffectKind, EffectLog, EffectLogBuilder, InMemoryStore, ToolBuilder};

#[cfg(feature = "sqlite")]
use effect_log_core::SqliteStore;

/// Python-visible enum for EffectKind.
#[pyclass(name = "EffectKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
enum PyEffectKind {
    ReadOnly = 0,
    IdempotentWrite = 1,
    Compensatable = 2,
    IrreversibleWrite = 3,
    ReadThenWrite = 4,
}

impl From<PyEffectKind> for EffectKind {
    fn from(kind: PyEffectKind) -> Self {
        match kind {
            PyEffectKind::ReadOnly => EffectKind::ReadOnly,
            PyEffectKind::IdempotentWrite => EffectKind::IdempotentWrite,
            PyEffectKind::Compensatable => EffectKind::Compensatable,
            PyEffectKind::IrreversibleWrite => EffectKind::IrreversibleWrite,
            PyEffectKind::ReadThenWrite => EffectKind::ReadThenWrite,
        }
    }
}

/// A registered tool descriptor for Python.
#[pyclass(name = "ToolDef")]
struct PyToolDef {
    name: String,
    effect_kind: PyEffectKind,
    func: Py<PyAny>,
    compensate: Option<Py<PyAny>>,
}

#[pymethods]
impl PyToolDef {
    #[new]
    #[pyo3(signature = (name, effect_kind, func, compensate=None))]
    fn new(
        name: String,
        effect_kind: PyEffectKind,
        func: Py<PyAny>,
        compensate: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            name,
            effect_kind,
            func,
            compensate,
        }
    }
}

/// Shared tokio runtime for all PyEffectLog instances.
///
/// Creating a new runtime per instance wastes OS threads if the user creates
/// many instances (e.g., per-request). A shared runtime avoids this.
fn shared_runtime() -> &'static tokio::runtime::Runtime {
    use std::sync::OnceLock;
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Runtime::new().expect("failed to create shared tokio runtime")
    })
}

/// The main Python API for effect-log.
#[pyclass(name = "EffectLog")]
struct PyEffectLog {
    inner: Arc<tokio::sync::Mutex<EffectLog>>,
    runtime: &'static tokio::runtime::Runtime,
}

#[pymethods]
impl PyEffectLog {
    /// Create a new EffectLog.
    ///
    /// Args:
    ///     execution_id: Unique ID for this execution run.
    ///     tools: List of ToolDef objects to register.
    ///     storage: Storage URI. "memory" for in-memory, "sqlite:///path" for SQLite.
    ///     recover: If True, recover from a prior execution.
    #[new]
    #[pyo3(signature = (execution_id, tools, storage="memory", recover=false))]
    fn new(
        py: Python<'_>,
        execution_id: String,
        tools: Vec<Bound<'_, PyToolDef>>,
        storage: &str,
        recover: bool,
    ) -> PyResult<Self> {
        let runtime = shared_runtime();

        let store: Arc<dyn EffectStore> = if storage == "memory" {
            Arc::new(InMemoryStore::new())
        } else if let Some(path) = storage.strip_prefix("sqlite:///") {
            #[cfg(feature = "sqlite")]
            {
                let path = path.to_string();
                runtime
                    .block_on(async {
                        SqliteStore::open(&path)
                            .await
                            .map(|s| Arc::new(s) as Arc<dyn EffectStore>)
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            #[cfg(not(feature = "sqlite"))]
            {
                let _ = path;
                return Err(PyRuntimeError::new_err(
                    "SQLite support not compiled in. Rebuild with 'sqlite' feature.",
                ));
            }
        } else {
            return Err(PyRuntimeError::new_err(format!(
                "Unsupported storage: {storage}. Use 'memory' or 'sqlite:///path'"
            )));
        };

        let mut builder = EffectLogBuilder::new()
            .store(store)
            .execution_id(execution_id);

        for tool_bound in &tools {
            let tool_def = tool_bound.borrow();
            let kind: EffectKind = tool_def.effect_kind.into();
            let func = tool_def.func.clone_ref(py);
            let compensate = tool_def.compensate.as_ref().map(|c| c.clone_ref(py));
            let tool_name = tool_def.name.clone();

            let tool_fn: effect_log_core::registry::ToolFn = Arc::new(move |input| {
                let func = Python::with_gil(|py| func.clone_ref(py));
                Box::pin(async move {
                    let result: Result<serde_json::Value, String> = Python::with_gil(|py| {
                        let input_str = serde_json::to_string(&input).unwrap_or_default();
                        let py_input = py
                            .import("json")
                            .and_then(|json| json.call_method1("loads", (input_str,)))?;
                        let result = func.call1(py, (py_input,))?;
                        let result_str: String = py
                            .import("json")
                            .and_then(|json| json.call_method1("dumps", (result,)))
                            .and_then(|s| s.extract())?;
                        serde_json::from_str(&result_str)
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
                    })
                    .map_err(|e: PyErr| e.to_string());

                    match result {
                        Ok(value) => value,
                        Err(error) => {
                            // Propagate the error details so the executor can see what happened
                            serde_json::json!({
                                "__effect_log_error": true,
                                "error": error,
                            })
                        }
                    }
                })
            });

            let mut tool_builder = ToolBuilder::new(&tool_name, kind).func(tool_fn);

            if let Some(comp_py) = compensate {
                let compensate_fn: effect_log_core::registry::CompensateFn =
                    Arc::new(move |input| {
                        let comp_fn = Python::with_gil(|py| comp_py.clone_ref(py));
                        Box::pin(async move {
                            Python::with_gil(|py| {
                                let input_str = serde_json::to_string(&input).unwrap_or_default();
                                let py_input = py
                                    .import("json")
                                    .and_then(|json| json.call_method1("loads", (input_str,)))?;
                                comp_fn.call1(py, (py_input,))?;
                                Ok::<_, PyErr>(())
                            })
                            .map_err(|e| e.to_string())
                        })
                    });
                tool_builder = tool_builder.compensate(compensate_fn);
            }

            let tool = tool_builder
                .build()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            builder = builder
                .register_tool(tool)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }

        let inner = if recover {
            runtime
                .block_on(builder.recover())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else {
            builder
                .build()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        };

        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(inner)),
            runtime,
        })
    }

    /// Execute a tool call through the effect log.
    fn execute(
        &self,
        py: Python<'_>,
        tool_name: String,
        args: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let json_mod = py.import("json")?;
        let args_str: String = json_mod.call_method1("dumps", (args,))?.extract()?;
        let input: serde_json::Value =
            serde_json::from_str(&args_str).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let inner = Arc::clone(&self.inner);
        let runtime = self.runtime;

        // Release the GIL before block_on: the tool function will need to
        // re-acquire it via Python::with_gil, so holding it here deadlocks.
        let result = py.allow_threads(|| {
            runtime.block_on(async {
                let log = inner.lock().await;
                log.execute(&tool_name, input).await
            })
        });

        match result {
            Ok(value) => {
                // Check if the tool function returned an error marker
                if value.get("__effect_log_error").and_then(|v| v.as_bool()) == Some(true) {
                    let error_msg = value
                        .get("error")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown python tool error");
                    return Err(PyRuntimeError::new_err(format!(
                        "Tool '{tool_name}' failed: {error_msg}"
                    )));
                }
                let result_str = serde_json::to_string(&value)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let py_result = json_mod.call_method1("loads", (result_str,))?;
                Ok(py_result.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the execution history.
    fn history(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = Arc::clone(&self.inner);
        let runtime = self.runtime;
        let result = py.allow_threads(|| {
            runtime.block_on(async {
                let log = inner.lock().await;
                log.history().await
            })
        });

        match result {
            Ok(entries) => {
                let json_mod = py.import("json")?;
                let list = pyo3::types::PyList::empty(py);
                for (intent, completion) in entries {
                    let dict = PyDict::new(py);
                    dict.set_item("tool_name", &intent.tool_name)?;
                    dict.set_item("effect_kind", format!("{:?}", intent.effect_kind))?;
                    dict.set_item("sequence", intent.cursor.sequence_number)?;

                    let input_str = serde_json::to_string(&intent.input)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    let py_input = json_mod.call_method1("loads", (input_str,))?;
                    dict.set_item("input", py_input)?;

                    if let Some(c) = completion {
                        dict.set_item("outcome", format!("{:?}", c.outcome))?;
                        let resp_str = serde_json::to_string(&c.sealed_response)
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                        let py_resp = json_mod.call_method1("loads", (resp_str,))?;
                        dict.set_item("sealed_response", py_resp)?;
                    } else {
                        dict.set_item("outcome", "Incomplete")?;
                        dict.set_item("sealed_response", py.None())?;
                    }

                    list.append(dict)?;
                }
                Ok(list.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }
}

/// Python module definition.
#[pymodule]
fn effect_log_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEffectKind>()?;
    m.add_class::<PyToolDef>()?;
    m.add_class::<PyEffectLog>()?;
    Ok(())
}
