use crate::pcn::PCN;
use rand::Rng;

pub struct Data {
    patterns: Vec<f64>,
}

pub struct Evaluation {
    patterns: Vec<f64>,
}

trait PCNAlgorithms {
    fn train_supervised<I: IntoIterator<Item = Data>, R: Rng>(&mut self, rng: &mut R, batch: I, alpha: f64, gamma: f64, inference_steps: usize);
    fn train_unsupervised<I: IntoIterator<Item = Data>>(&mut self, batch: I);
    fn evaluate<I: IntoIterator<Item = Evaluation>, R: Rng>(&mut self, rng: &mut R, batch: I, gamma: f64, inference_steps: usize) -> f64;
}

impl<NodeId: Eq + Ord + Clone> PCNAlgorithms for PCN<NodeId> {
    fn train_supervised<I: IntoIterator<Item = Data>, R: Rng>(&mut self, rng: &mut R, batch: I, alpha: f64, gamma: f64, inference_steps: usize) {
        for pattern in batch {
            let pattern = pattern.patterns;
            self.randomize_values(rng);
            // set node types
            // set values for nodes

            self.inference_steps(gamma, inference_steps);
            self.learn_hebb(alpha);
        }
    }

    fn train_unsupervised<I: IntoIterator<Item = Data>>(&mut self, batch: I) {
        todo!()
    }

    fn evaluate<I: IntoIterator<Item = Evaluation>, R: Rng>(&mut self, rng: &mut R, batch: I, gamma: f64, inference_steps: usize) -> f64 {
        let mut err_sum = 0.;

        for pattern in batch {
            self.randomize_values(rng);
            // set node types
            // set values for nodes

            self.inference_steps(gamma, inference_steps);
            // get output pattern
            // get square error
            // accumulate error
        }

        err_sum
    }
}
