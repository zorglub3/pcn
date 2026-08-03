use serde::{Deserialize, Serialize};
use crate::dvector::eval_inplace;
use crate::dvector::max_f64;

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFn {
    Tanh,
    Logistic,
    ReLu,
    LeakyReLu(f64),
    SoftPlus,
    SoftMax,
    Linear,
}

impl ActivationFn {
    pub fn eval_inplace(&self, values: &mut [f64]) {
        use ActivationFn::*;

        match self {
            Tanh => eval_inplace(|v| v.tanh(), values),
            Logistic => eval_inplace(|v| 1. / (1. + (-v).exp()), values),
            ReLu => eval_inplace(|v| v.max(0.), values),
            LeakyReLu(a) => eval_inplace(|v| v.max(*a * v), values),
            SoftPlus => eval_inplace(|v| (1. + v.exp()).ln(), values),
            SoftMax => {
                if values.len() > 0 {
                    let d = max_f64(values).unwrap_or(0.);
                    let h: f64 = values.iter().map(|v| (v - d).exp()).sum();
                    eval_inplace(|v| (v - d).exp() / h, values);
                }
            }
            Linear => { /* do nothing */ },
        }
    }

    pub fn diff_inplace_mul(&self, multiplier: &[f64], values: &mut [f64]) {
        debug_assert_eq!(multiplier.len(), values.len());

        use ActivationFn::*;

        match self {
            SoftMax => {
                let mut s = vec![0.; values.len()];
                s.clone_from_slice(values);
                self.diff_inplace(&mut s);

                for i in 0..values.len() {
                    let mut acc = 0.;
                    for j in 0..values.len() {
                        if i == j {
                            acc += s[i] * (1. - s[i]);
                        } else {
                            acc -= s[i] * s[j];
                        }
                    }
                    values[i] = acc;
                }
            }
            _ => {
                self.diff_inplace(values);
                for (v, m) in values.iter_mut().zip(multiplier) {
                    *v *= m;
                }
            }
        }
    }

    pub fn diff_inplace(&self, values: &mut [f64]) {
        use ActivationFn::*;

        match self {
            Tanh => eval_inplace(|v| { let x = v.tanh(); 1. - x * x }, values),
            Logistic => eval_inplace(|v| { let x = 1. / (1. + (-v).exp()); x * (1. - x) }, values),
            ReLu => eval_inplace(|v| if v < 0. { 0. } else { 1. }, values),
            LeakyReLu(a) => eval_inplace(|v| if v < a * v { *a } else { 1. }, values),
            SoftPlus => eval_inplace(|v| 1. / (1. + (-v).exp()), values),
            SoftMax => todo!(),
            Linear => values.fill(1.),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tanh() {
        let f = ActivationFn::Tanh;

        let values = [1., -2., 0.];

        let mut output = values.clone();
        f.eval_inplace(&mut output);
        assert_eq!(output[0], 1.0_f64.tanh());
        assert_eq!(output[1], (-2.0_f64).tanh());
        assert_eq!(output[2], 0.0);

        let mut output = values.clone();
        f.diff_inplace(&mut output);
        assert_eq!(output[0], 1. - 1.0_f64.tanh() * 1.0_f64.tanh());
        assert_eq!(output[1], 1. - (-2.0_f64).tanh() * (-2.0_f64).tanh());
        assert_eq!(output[2], 1.);
    }

    #[test]
    fn test_softmax() {
        let f = ActivationFn::SoftMax;

        let values = [1., 0., -1.];
        let mut output = values.clone();
        f.eval_inplace(&mut output);
        assert_eq!(output[0], 0.6652409557748219);
        assert_eq!(output[1], 0.24472847105479767);
        assert_eq!(output[2], 9.003057317038046e-2);

        // TODO diff_inplace_mul
    }
}
