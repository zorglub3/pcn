use serde::{Deserialize, Serialize};

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFn {
    Tanh,
    Logistic,
    ReLu,
    LeakyReLu(f64),
    SoftPlus,
    Linear,
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
            Logistic => {
                for i in 0..input.len() {
                    output[i] = 1. / (1. + (-input[i]).exp());
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
            Linear => {
                output.copy_from_slice(input);
            }
        }
    }

    pub fn eval_mul(&self, input: &[f64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), output.len());

        use ActivationFn::*;

        match self {
            Tanh => {
                for i in 0..input.len() {
                    output[i] *= input[i].tanh();
                }
            }
            Logistic => {
                for i in 0..input.len() {
                    output[i] *= 1. / (1. + (-input[i]).exp());
                }
            }
            ReLu => {
                for i in 0..input.len() {
                    output[i] *= input[i].max(0.);
                }
            }
            LeakyReLu(a) => {
                for i in 0..input.len() {
                    output[i] *= input[i].max(a * input[i]);
                }
            }
            SoftPlus => {
                for i in 0..input.len() {
                    output[i] *= (1. + input[i].exp()).ln();
                }
            }
            Linear => {
                for i in 0..input.len() {
                    output[i] *= input[i];
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
            Logistic => {
                for i in 0..input.len() {
                    let d = 1. / (1. + (-input[i]).exp());
                    output[i] = d * (1. - d);
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
            Linear => {
                output.fill(1.);
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
            Logistic => {
                for i in 0..input.len() {
                    let d = 1. / (1. + (-input[i]).exp());
                    output[i] *= d * (1. - d);
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
            Linear => {} // no effect
        }
    }

    pub fn diff2(&self, input: &[f64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), output.len());

        use ActivationFn::*;

        match self {
            Tanh => {
                for i in 0..input.len() {
                    let v = input[i].tanh();
                    let d = 1. - v * v;
                    output[i] = -2. * d * v;
                }
            }
            _ => unimplemented!("coming soon? ever..?"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tanh() {
        let f = ActivationFn::Tanh;

        let input = [1., -2., 0.];

        let mut output = [0.; 3];

        f.eval(&input, &mut output);
        assert_eq!(output[0], 1.0_f64.tanh());
        assert_eq!(output[1], (-2.0_f64).tanh());
        assert_eq!(output[2], 0.0);

        let mut output = [1., -1., 0.];
        f.eval_mul(&input, &mut output);
        assert_eq!(output[0], 1.0_f64.tanh());
        assert_eq!(output[1], -1. * (-2.0_f64).tanh());
        assert_eq!(output[2], 0.0);

        let mut output = [1., -1., 0.];
        f.diff(&input, &mut output);
        assert_eq!(output[0], 1. - 1.0_f64.tanh() * 1.0_f64.tanh());
        assert_eq!(output[1], 1. - (-2.0_f64).tanh() * (-2.0_f64).tanh());
        assert_eq!(output[2], 1.);

        let mut output = [1., -1., 0.];
        f.diff_mul(&input, &mut output);
        assert_eq!(output[0], 1. - 1.0_f64.tanh() * 1.0_f64.tanh());
        assert_eq!(output[1], -1. + (-2.0_f64).tanh() * (-2.0_f64).tanh());
        assert_eq!(output[2], 0.);
    }
}
