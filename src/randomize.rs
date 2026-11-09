use crate::activation::ActivationFn;
use crate::activation::FloatActivationFn;
use crate::pcn::Edge;
use crate::pcn::PCN;
use num_traits::float::Float;
use rand::Rng;
use rand::RngExt;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, Mul, MulAssign, SubAssign};

pub trait PCNDistribution<N> {
    fn next<R: Rng>(&self, rng: &mut R) -> N;
}

pub trait RandomizeDistribution<N>: ActivationFn<N> {
    type D: PCNDistribution<N>;

    fn weight_distribution(self, rows: usize, columns: usize) -> Self::D;
}

pub enum FloatDistribution<N: Float> {
    Uniform { scale: N },
    Normal { expected: N, deviation: N },
}

impl PCNDistribution<f64> for FloatDistribution<f64> {
    fn next<R: Rng>(&self, rng: &mut R) -> f64 {
        match self {
            Self::Uniform { scale } => *scale * rng.random::<f64>(),
            Self::Normal { .. } => todo!("normal distribution for floats"),
        }
    }
}

impl RandomizeDistribution<f64> for FloatActivationFn<f64> {
    type D = FloatDistribution<f64>;

    fn weight_distribution(self, rows: usize, columns: usize) -> Self::D {
        match self {
            FloatActivationFn::Tanh(_) => {
                // Xavier uniform
                let scale = (6. / ((rows + columns) as f64)).sqrt();
                FloatDistribution::Uniform { scale }
            }
            FloatActivationFn::Logistic(_) => todo!(), // Xavier again?
            FloatActivationFn::ReLu(_) => todo!(),
            FloatActivationFn::Linear(_) => todo!(), // just uniform around 0
            _ => todo!(),
        }
    }
}

pub trait RandomizePCN<N, A: ActivationFn<N>> {
    fn randomize_weights<R: Rng>(&mut self, rng: &mut R);
    fn randomize_values<R: Rng>(&mut self, rng: &mut R);
}

impl<Id, N, A> RandomizePCN<N, A> for PCN<N, A, Id>
where
    Id: Eq + Ord + Clone,
    N: MulAssign + Mul<Output = N> + Default + AddAssign + SubAssign + Sum + Copy + Debug,
    A: ActivationFn<N> + RandomizeDistribution<N> + Copy,
{
    fn randomize_weights<R: Rng>(&mut self, rng: &mut R) {
        for Edge {
            source,
            target,
            weight_matrix_index,
        } in self.edges.iter()
        {
            let source_size = self.node_sizes[*source];
            let target_size = self.node_sizes[*target];

            let distr =
                self.activation_functions[*target].weight_distribution(target_size, source_size);

            let matrix = &mut self.weight_matrices[*weight_matrix_index];

            matrix.generate(|_r, _c| distr.next(rng));
        }
    }

    fn randomize_values<R: Rng>(&mut self, _rng: &mut R) {
        todo!()
    }
}
