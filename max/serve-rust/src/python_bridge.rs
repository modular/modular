use pyo3::prelude::*;

pub struct PythonBridge {
    decode_fn: PyObject,
}

impl PythonBridge {
    pub fn new() -> Self {
        Python::with_gil(|py| {
            let max_serve = py
                .import_bound("max.serve")
                .expect("Failed to import max.serve Python module");
            let decode_fn = max_serve
                .getattr("decode")
                .expect("Failed to find 'decode' function in max.serve")
                .into();
            Self { decode_fn }
        })
    }

    pub fn decode_tokens(&self, tokens: Vec<i32>) -> PyResult<String> {
        Python::with_gil(|py| self.decode_fn.call1(py, (tokens,))?.extract(py))
    }
}
