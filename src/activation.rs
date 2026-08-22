// use crate::dvector::eval_inplace;
// use crate::dvector::max_f64;
use serde::{Deserialize, Serialize};
use num_traits::float::Float;
use num_traits::identities::{one, zero};
use std::marker::PhantomData;

pub trait ActivationFn<N> {
    fn eval_inplace(&self, values: &mut [N]);
    fn diff_inplace_mul(&self, multiplier: &[N], values: &mut [N]);
    fn diff_inplace(&self, values: &mut [N]);
}

fn eval_inplace<N, F: Fn(N) -> N>(f: F, a: &mut [N]) {
    for v in a {
        *v = f(*v);
    }
}

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum FloatActivationFn<N: Float> {
    Tanh(PhantomData<N>),
    Logistic(PhantomData<N>),
    ReLu(PhantomData<N>),
    LeakyReLu(N, PhantomData<N>),
    SoftPlus(PhantomData<N>),
    SoftMax(PhantomData<N>),
    Linear(PhantomData<N>),
}

impl<N: Float> FloatActivationFn<N> {
    pub const TANH: Self = FloatActivationFn::Tanh(PhantomData);
    pub const LOGISTIC: Self = FloatActivationFn::Logistic(PhantomData);
    pub const RELU: Self = FloatActivationFn::ReLu(PhantomData);
    pub const SOFTPLUS: Self = FloatActivationFn::SoftPlus(PhantomData);
    pub const SOFTMAX: Self = FloatActivationFn::SoftMax(PhantomData);
    pub const LINEAR: Self = FloatActivationFn::Linear(PhantomData);

    pub fn leaky_relu(x: N) -> Self {
        FloatActivationFn::LeakyReLu(x)
    }
}

impl<N: Float> ActivationFn<N> for FloatActivationFn<N> {
    fn eval_inplace(&self, values: &mut [N]) {
        use FloatActivationFn::*;

        match self {
            Tanh(_) => eval_inplace(|v| v.tanh(), values),
            Logistic(_) => eval_inplace(|v| one() / (one() + (-v).exp()), values),
            ReLu(_) => eval_inplace(|v| v.max(zero), values),
            LeakyReLu(a, _) => eval_inplace(|v| v.max(*a * v), values),
            SoftPlus(_) => eval_inplace(|v| (one() + v.exp()).ln(), values),
            SoftMax(_) => {
                if values.len() > 0 {
                    let d = max_f64(values).unwrap_or(zero());
                    let h: f64 = values.iter().map(|v| (v - d).exp()).sum();
                    eval_inplace(|v| (v - d).exp() / h, values);
                }
            }
            Linear(_) => { /* do nothing */ }
        }
    }

    fn diff_inplace_mul(&self, multiplier: &[N], values: &mut [N]) {
        debug_assert_eq!(multiplier.len(), values.len());

        use FloatActivationFn::*;

        match self {
            SoftMax(_) => {
                let mut s = vec![0.; values.len()];
                s.clone_from_slice(values);
                self.diff_inplace(&mut s);

                for i in 0..values.len() {
                    let mut acc = 0.;
                    for j in 0..values.len() {
                        if i == j {
                            acc += s[i] * (one() - s[i]);
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

    fn diff_inplace(&self, values: &mut [N]) {
        use FloatActivationFn::*;

        match self {
            Tanh(_) => eval_inplace(
                |v| {
                    let x = v.tanh();
                    one() - x * x
                },
                values,
            ),
            Logistic(_) => eval_inplace(
                |v| {
                    let x = one() / (one() + (-v).exp());
                    x * (one() - x)
                },
                values,
            ),
            ReLu(_) => eval_inplace(|v| if v < zero() { zero() } else { one() }, values),
            LeakyReLu(a, _) => eval_inplace(|v| if v < a * v { *a } else { one() }, values),
            SoftPlus(_) => eval_inplace(|v| one() / (one() + (-v).exp()), values),
            SoftMax(_) => unimplemented!("use the diff_inplace_mul function"),
            Linear(_) => values.fill(one()),
        }
    }
}

/*
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
*/
