use crate::activation::ActivationFn;
use crate::dmatrix::DMatrix;
use crate::dvector::hadamard_inplace;
use crate::dvector::randomize_vec;
use crate::dvector::scale_add_inplace;
use crate::dvector::sub_inplace;

pub struct Network {
    pub x: Vec<Vec<f64>>,
    pub u: Vec<Vec<f64>>,
    pub a: Vec<Vec<f64>>,
    pub e: Vec<Vec<f64>>,
    pub w: Vec<DMatrix<f64>>,
    pub size: usize,
}

impl Network {
    pub fn new(dims: &[usize]) -> Self {
        let mut x = Vec::new();
        let mut u = Vec::new();
        let mut a = Vec::new();
        let mut e = Vec::new();
        let mut w = Vec::new();
        let mut size = 0;

        let mut opt_x_len = None;

        for d in dims {
            x.push(vec![0.; *d]);
            u.push(vec![0.; *d]);
            a.push(vec![0.; *d]);
            e.push(vec![0.; *d]);

            size += 1;

            if let Some(prev_x_len) = opt_x_len {
                w.push(DMatrix::new(prev_x_len, *d, 0.));
            }

            opt_x_len = Some(*d);
        }

        Self {
            x,
            u,
            a,
            e,
            w,
            size,
        }
    }

    pub fn inference(&mut self, gamma: f64) {
        for i in 0..self.size - 1 {
            self.w[i].mul_vec(&self.x[i + 1], &mut self.a[i]);
        }

        for i in 0..self.size {
            ActivationFn::Tanh.eval(&self.a[i], &mut self.u[i]);
        }

        let mut err_sum_sqr = 0.;
        for i in 0..self.size {
            for j in 0..self.x[i].len() {
                let err = self.x[i][j] - self.u[i][j];
                self.e[i][j] = err;
                err_sum_sqr += err * err;
            }
        }

        err_sum_sqr /= 2.;
        println!("energy: {}", err_sum_sqr);

        for i in 1..self.size {
            let mut temp = vec![0.; self.x[i - 1].len()];
            let mut acc = vec![0.; self.x[i].len()];
            ActivationFn::Tanh.diff(&self.a[i - 1], &mut temp);
            hadamard_inplace(&self.e[i - 1], &mut temp);
            self.w[i - 1].trans_mul_vec_add(&temp, &mut acc);

            sub_inplace(&self.e[i], &mut acc);
            scale_add_inplace(gamma, &acc, &mut self.x[i]);
        }
    }

    pub fn randomize_weights(&mut self, rng: &mut impl rand::Rng) {
        for i in 0..self.size - 1 {
            self.w[i].randomize_xavier(rng);
        }
    }

    pub fn reset(&mut self, rng: &mut impl rand::Rng) {
        for i in 0..self.size {
            randomize_vec(0.5, &mut self.x[i], rng);
            self.u[i].fill(0.);
            self.e[i].fill(0.);
            self.a[i].fill(0.);
        }
    }

    pub fn set_input(&mut self, input: &[f64]) {
        debug_assert!(self.size > 0);
        debug_assert_eq!(self.x[0].len(), input.len());

        self.x[0].copy_from_slice(input);
    }
}
