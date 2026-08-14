use crate::pcn::PCN;
use rand::Rng;
use std::collections::BTreeMap;

pub struct Data<Id: Eq + Ord + Clone> {
    sensor_patterns: Vec<(Id, Vec<f64>)>,
}

pub struct LabeledData<Id: Eq + Ord + Clone> {
    sensor_patterns: Vec<(Id, Vec<f64>)>,
    label_patterns: Vec<(Id, Vec<f64>)>,
}

trait PCNAlgorithms<Id: Eq + Ord + Clone> {
    fn train_supervised<I: IntoIterator<Item = LabeledData<Id>>, R: Rng>(
        &mut self,
        rng: &mut R,
        batch: I,
        alpha: f64,
        gamma: f64,
        inference_steps: usize,
    );
    fn train_unsupervised<I: IntoIterator<Item = Data<Id>>>(&mut self, batch: I);
    fn evaluate<I: IntoIterator<Item = LabeledData<Id>>, R: Rng>(
        &mut self,
        rng: &mut R,
        batch: I,
        gamma: f64,
        inference_steps: usize,
    ) -> f64;
    fn infer<R: Rng>(
        &mut self,
        gamma: f64,
        inference_steps: usize,
        data: Data<Id>,
        output: &mut BTreeMap<Id, Vec<f64>>,
    );
    fn new_output_map<I: IntoIterator<Item = Id>>(&self, output_nodes: I)
        -> BTreeMap<Id, Vec<f64>>;
}

impl<Id: Eq + Ord + Clone> PCNAlgorithms<Id> for PCN<Id> {
    fn train_supervised<I: IntoIterator<Item = LabeledData<Id>>, R: Rng>(
        &mut self,
        rng: &mut R,
        batch: I,
        alpha: f64,
        gamma: f64,
        inference_steps: usize,
    ) {
        for pattern in batch {
            self.randomize_values(rng);
            // set node types
            // set values for nodes

            self.inference_steps(gamma, inference_steps);
            self.learn_hebb(alpha);
        }
    }

    fn train_unsupervised<I: IntoIterator<Item = Data<Id>>>(&mut self, batch: I) {
        todo!()
    }

    fn evaluate<I: IntoIterator<Item = LabeledData<Id>>, R: Rng>(
        &mut self,
        rng: &mut R,
        batch: I,
        gamma: f64,
        inference_steps: usize,
    ) -> f64 {
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

    fn infer<R: Rng>(
        &mut self,
        gamma: f64,
        inference_steps: usize,
        data: Data<Id>,
        output: &mut BTreeMap<Id, Vec<f64>>,
    ) {
        // randomize network values (not weights)

        for sensor_pattern in data.sensor_patterns.iter() {
            // set values and type for nodes
            //
        }

        self.inference_steps(gamma, inference_steps);

        for (key, value) in output.iter_mut() {
            // set values in map
        }
    }

    fn new_output_map<I: IntoIterator<Item = Id>>(
        &self,
        output_nodes: I,
    ) -> BTreeMap<Id, Vec<f64>> {
        let mut output_map = BTreeMap::new();

        for id in output_nodes {
            // insert dummy vectors in output_map
        }

        output_map
    }
}
