use serde::{Serialize, Deserialize};

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFn {
    Tanh,
    ReLu,
    LeakyReLu(f64),
    SoftPlus,
}

impl ActivationFn {
    pub fn eval(&self, input: &[f64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), output.len());

        use ActivationFn::*;

        match self {
            Tanh => {
                for i in 0..input.len() {
                    output[i] = input[i].tanh();
                }
            }
            ReLu => {
                for i in 0..input.len() {
                    output[i] = input[i].max(0.);
                }
            }
            LeakyReLu(a) => {
                for i in 0..input.len() {
                    output[i] = input[i].max(a * input[i]);
                }
            }
            SoftPlus => {
                for i in 0..input.len() {
                    output[i] = (1. + input[i].exp()).ln();
                }
            }
        }
    }

    pub fn diff(&self, input: &[f64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), output.len());

        use ActivationFn::*;

        match self {
            Tanh => {
                for i in 0..input.len() {
                    let v = input[i].tanh();
                    output[i] = 1. - v * v;
                }
            }
            ReLu => {
                for i in 0..input.len() {
                    output[i] = if input[i] < 0. { 0. } else { 1. };
                }
            }
            LeakyReLu(a) => {
                for i in 0..input.len() {
                    output[i] = if input[i] < a * input[i] { *a } else { 1. };
                }
            }
            SoftPlus => {
                for i in 0..input.len() {
                    output[i] = 1. / (1. + (-input[i]).exp());
                }
            }
        }
    }

    pub fn diff_mul(&self, input: &[f64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), output.len());

        use ActivationFn::*;

        match self {
            Tanh => {
                for i in 0..input.len() {
                    let v = input[i].tanh();
                    output[i] *= 1. - v * v;
                }
            }
            ReLu => {
                for i in 0..input.len() {
                    output[i] *= if input[i] < 0. { 0. } else { 1. };
                }
            }
            LeakyReLu(a) => {
                for i in 0..input.len() {
                    output[i] *= if input[i] < a * input[i] { *a } else { 1. };
                }
            }
            SoftPlus => {
                for i in 0..input.len() {
                    output[i] *= 1. / (1. + (-input[i]).exp());
                }
            }
        }
    }
}
