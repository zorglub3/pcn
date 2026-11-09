use std::collections::BTreeMap;
use crate::pcn::PCN;
use crate::pcn::NodeType;
use crate::activation::ActivationFn;
use std::iter::Sum;
use std::fmt::Debug;
use std::ops::{AddAssign, Mul, MulAssign, Sub, SubAssign};

pub struct Data<Id: Eq + Ord + Clone, N> {
    pub sensor_patterns: Vec<(Id, Vec<N>)>,
}

/*
impl<Id: Eq + Ord + Clone> Data<Id, bool> {
    pub fn map_values<N>(&self, true_val: N, false_val: N) -> Data<Id, N> {
        Data {
            sensor_patterns: todo!() 
        }
    }
}
*/

pub struct LabeledData<Id: Eq + Ord + Clone, N> {
    pub sensor_patterns: Vec<(Id, Vec<N>)>,
    pub label_patterns: Vec<(Id, Vec<N>)>,
}

/*
impl<Id: Eq + Ord + Clone> LabeledData<Id, bool> {
    pub fn map_values<N>(&self, true_val: N, false_val: N) -> LabeledData<Id, N> {
        LabeledData {
            sensor_patterns: todo!(),
            label_patterns: todo!(),
        }
    }
} 
*/

pub trait PCNAlgorithms<Id: Eq + Ord + Clone, N, A>
where N: MulAssign + Mul<Output = N> + Default + AddAssign + SubAssign + Sum + Copy,
      A: ActivationFn<N>,
{
    fn train_supervised<K: Into<N> + Clone>(
        &mut self,
        batch: &[LabeledData<Id, N>],
        alpha: K,
        gamma: K,
        inference_steps: usize,
    );
    fn train_unsupervised<K: Into<N> + Clone>(
        &mut self, 
        batch: &[Data<Id, N>],
        alpha: K,
        gamma: K,
        inference_steps: usize,
    );
    fn evaluate<K: Into<N> + Clone>(
        &mut self,
        batch: &[LabeledData<Id, N>],
        gamma: K,
        inference_steps: usize,
    ) -> f64;
    fn infer<K: Into<N> + Clone>(
        &mut self,
        gamma: K,
        inference_steps: usize,
        data: &Data<Id, N>,
        output: &mut BTreeMap<Id, Vec<N>>,
    );
    fn new_output_map<I: IntoIterator<Item = Id>>(&self, output_nodes: I)
        -> BTreeMap<Id, Vec<N>>;
}

fn set_sensor_data<N, A, Id>(pcn: &mut PCN<N, A, Id>, sensor_data: &Data<Id, N>) 
where N: Sub<Output = N> + MulAssign + Mul<Output = N> + Default + AddAssign + Sum + Copy + Debug + SubAssign, A: ActivationFn<N>,
      A: ActivationFn<N>,
      Id: Eq + Ord + Clone,
{
    for (node_id, values) in sensor_data.sensor_patterns.iter() {
        pcn.set_node_type(node_id, NodeType::Sensor);
        pcn.set_predictions(node_id, values);
    }
}

fn set_labeled_data<N, A, Id>(pcn: &mut PCN<N, A, Id>, labeled_data: &LabeledData<Id, N>)
where N: Sub<Output = N> + MulAssign + Mul<Output = N> + Default + AddAssign + Sum + Copy + Debug + SubAssign, A: ActivationFn<N>,
      A: ActivationFn<N>,
      Id: Eq + Ord + Clone,
{
    for (node_id, values) in labeled_data.sensor_patterns.iter() {
        pcn.set_node_type(node_id, NodeType::Sensor);
        pcn.set_predictions(node_id, values);
    }

    for (node_id, values) in labeled_data.label_patterns.iter() {
        pcn.set_node_type(node_id, NodeType::Label);
        pcn.set_values(node_id, values);
    }
}

impl<N, A, Id> PCNAlgorithms<Id, N, A> for PCN<N, A, Id>
where N: Sub<Output = N> + MulAssign + Mul<Output = N> + Default + AddAssign + Sum + Copy + Debug + SubAssign, A: ActivationFn<N>,
      A: ActivationFn<N>,
      Id: Eq + Ord + Clone,
{
    fn train_supervised<K: Into<N> + Clone>(
        &mut self,
        batch: &[LabeledData<Id, N>],
        alpha: K,
        gamma: K,
        inference_steps: usize,
    ) {
        self.clear_node_labels();

        for labeled_data in batch.iter() {
            set_labeled_data(self, &labeled_data);
            self.inference_steps(gamma.clone(), inference_steps);
            self.learn_hebb(alpha.clone());
        }
    }

    fn train_unsupervised<K: Into<N> + Clone>(
        &mut self, 
        batch: &[Data<Id, N>],
        alpha: K,
        gamma: K,
        inference_steps: usize,
    ) {
        self.clear_node_labels();

        for sensor_data in batch.into_iter() {
            set_sensor_data(self, &sensor_data);
            self.inference_steps(gamma.clone(), inference_steps);
            self.learn_hebb(alpha.clone());
        }
    }

    fn evaluate<K: Into<N> + Clone>(
        &mut self,
        batch: &[LabeledData<Id, N>],
        gamma: K,
        inference_steps: usize,
    ) -> f64 {
        self.clear_node_labels();

        let error_value = 0.;

        for labeled_data in batch.into_iter() {
            set_labeled_data(self, labeled_data);
            self.inference_steps(gamma.clone(), inference_steps);
            todo!("increment error");
        }

        error_value
    }

    fn infer<K: Into<N> + Clone>(
        &mut self,
        gamma: K,
        inference_steps: usize,
        data: &Data<Id, N>,
        output: &mut BTreeMap<Id, Vec<N>>,
    ) {
        self.clear_node_labels();

        set_sensor_data(self, &data);
        self.inference_steps(gamma, inference_steps);
        todo!("update output vectors");
    }

    fn new_output_map<I: IntoIterator<Item = Id>>(
        &self, 
        output_nodes: I,
    ) -> BTreeMap<Id, Vec<N>> {
        todo!("Create dummy vectors in output_nodes")
    }
}
