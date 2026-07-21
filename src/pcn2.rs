//! Implementation fo Predictive Coding Network. Loosely based on
//! "Introduction to Predictive Coding Networks for Machine Learning"
//! by Mikko Stenlund.

use crate::activation::ActivationFn;
use crate::dmatrix::DMatrix;
use crate::dvector::hadamard_inplace;
use crate::dvector::randomize_vec;
use crate::dvector::scale_sub_inplace;
use rand::Rng;
use std::collections::BTreeMap;

pub struct PCN<NodeId: Eq + Ord + Clone> {
    activation_functions: Vec<ActivationFn>,
    node_values: Vec<NodeValues>,
    node_predictions: Vec<NodePredictions>,
    node_gain_modulated_errors: Vec<NodePredictionDiffs>,
    node_errors: Vec<NodeErrors>,
    node_sizes: Vec<usize>,
    node_types: Vec<NodeType>,
    next_node_index: usize,
    weight_matrices: Vec<DMatrix<f64>>,
    edges: Vec<Edge>,
    nodes_map: BTreeMap<NodeId, NodeIndex>,
}

impl<NodeId: Eq + Ord + Clone> Default for PCN<NodeId> {
    fn default() -> Self {
        Self {
            activation_functions: Vec::new(),
            node_values: Vec::new(),
            node_predictions: Vec::new(),
            node_gain_modulated_errors: Vec::new(),
            node_errors: Vec::new(),
            node_sizes: Vec::new(),
            node_types: Vec::new(),
            next_node_index: 0,
            weight_matrices: Vec::new(),
            edges: Vec::new(),
            nodes_map: BTreeMap::new(),
        }
    }
}

impl<NodeId: Eq + Ord + Clone> PCN<NodeId> {
    pub fn add_node(&mut self, id: &NodeId, activation_function: ActivationFn, size: usize) {
        debug_assert!(!self.nodes_map.contains_key(&id));

        let node_index = self.next_node_index;
        self.next_node_index += 1;

        self.activation_functions.push(activation_function);
        self.node_values.push(NodeValues::new(size));
        self.node_predictions.push(NodePredictions::new(size));
        self.node_gain_modulated_errors
            .push(NodePredictionDiffs::new(size));
        self.node_errors.push(NodeErrors::new(size));
        self.node_sizes.push(size);
        self.node_types.push(Default::default());

        self.nodes_map.insert(id.clone(), node_index);
    }

    pub fn add_edge(&mut self, target_id: &NodeId, source_id: &NodeId) {
        debug_assert!(self.nodes_map.contains_key(source_id));
        debug_assert!(self.nodes_map.contains_key(target_id));

        let source = self.nodes_map.get(source_id).unwrap();
        let target = self.nodes_map.get(target_id).unwrap();

        let source_size = self.node_sizes[*source];
        let target_size = self.node_sizes[*target];
        let weight_matrix = DMatrix::new(target_size, source_size, 0.);
        let weight_matrix_index = self.weight_matrices.len();

        self.weight_matrices.push(weight_matrix);
        self.edges
            .push(Edge::new(*source, *target, weight_matrix_index));
    }

    // TODO use Xavier uniform distribution (or normal dist - find out which one fits)
    pub fn randomize_weights<R: Rng>(&mut self, rng: &mut R) {
        // Using uniform Xavier initialization. see
        // https://www.geeksforgeeks.org/deep-learning/xavier-initialization/

        for weight_matrix in self.weight_matrices.iter_mut() {
            let x = (6. / (weight_matrix.rows() + weight_matrix.cols()) as f64).sqrt();
            weight_matrix.randomize(x, rng);
        }
    }

    pub fn randomize_values<R: Rng>(&mut self, rng: &mut R) {
        for node_value in self.node_values.iter_mut() {
            randomize_vec(1., node_value.0.as_mut(), rng);
        }
    }

    pub fn compute_errors(&mut self) -> f64 {
        let mut error_square_sum = 0.;

        let iter = self
            .node_errors
            .iter_mut()
            .zip(&self.node_values)
            .zip(&self.node_predictions)
            .zip(&self.node_types);

        for (((error, value), prediction), node_type) in iter {
            let inner_iter = error
                .0
                .iter_mut()
                .zip(value.0.as_ref())
                .zip(prediction.0.as_ref());

            for ((e, v), p) in inner_iter {
                let err = if node_type.is_sensor() { p - v } else { v - p };
                *e = err;
                error_square_sum += err * err;
            }
        }

        error_square_sum
    }

    pub fn inference_steps(&mut self, gamma: f64, n: usize) -> f64 {
        let mut err = 0.;

        for _i in 0..n {
            self.compute_predictions();
            err = self.compute_errors();
            self.compute_values(gamma);
        }
        err
    }

    pub fn compute_predictions(&mut self) {
        // TODO Not just nodes of "Fixed" type should not update predictions
        //  Also nodes with no incoming edges - their predictions will otherwise be
        //  zero with no way to change.
        for (prediction, node_type) in self.node_predictions.iter_mut().zip(&self.node_types) {
            if node_type.update_predictions() {
                prediction.0.as_mut().fill(0.);
            }
        }

        for edge in self.edges.iter() {
            let matrix = &self.weight_matrices[edge.weight_matrix];
            let source = self.node_values[edge.source].0.as_ref();
            let target = self.node_predictions[edge.target].0.as_mut();

            if self.node_types[edge.target].update_predictions() {
                matrix.mul_vec_add(source, target);
            }
        }

        for (i, prediction) in self.node_predictions.iter_mut().enumerate() {
            if self.node_types[i].update_predictions() {
                self.activation_functions[i].eval_inplace(prediction.0.as_mut());
            }
        }
    }

    pub fn compute_gain_modulated_errors(&mut self) {
        for gain_modulated_errors in self.node_gain_modulated_errors.iter_mut() {
            gain_modulated_errors.0.as_mut().fill(0.);
        }

        for edge in self.edges.iter() {
            let matrix = &self.weight_matrices[edge.weight_matrix];
            let source = self.node_values[edge.source].0.as_ref();
            let target = self.node_gain_modulated_errors[edge.target].0.as_mut();

            matrix.mul_vec_add(source, target);
        }

        for (i, gain_modulated_errors) in self.node_gain_modulated_errors.iter_mut().enumerate() {
            self.activation_functions[i].diff_inplace(gain_modulated_errors.0.as_mut());
            hadamard_inplace(
                self.node_errors[i].0.as_ref(),
                gain_modulated_errors.0.as_mut(),
            );
        }
    }

    pub fn compute_values(&mut self, gamma: f64) {
        self.compute_gain_modulated_errors();

        for (e, v) in self.node_errors.iter().zip(self.node_values.iter_mut()) {
            scale_sub_inplace(gamma, e.0.as_ref(), v.0.as_mut());
        }

        for edge in self.edges.iter() {
            let w = &self.weight_matrices[edge.weight_matrix];
            let gme = self.node_gain_modulated_errors[edge.target].0.as_ref();
            let v = self.node_values[edge.source].0.as_mut();

            w.trans_mul_vec_add_scale(gamma, gme, v);
        }
    }

    pub fn learn_hebb(&mut self, alpha: f64) {
        self.compute_gain_modulated_errors();

        for edge in self.edges.iter() {
            let w = &mut self.weight_matrices[edge.weight_matrix];
            let h = &self.node_gain_modulated_errors[edge.target].0.as_ref();
            let x = &self.node_values[edge.source].0.as_ref();

            for r in w.rows_range() {
                for c in w.cols_range() {
                    w[(r, c)] += alpha * h[r] * x[c];
                }
            }
        }
    }

    pub fn learn_oja(&mut self, _alpha: f64) {
        todo!()
    }

    pub fn set_values(&mut self, node_id: &NodeId, values: &[f64]) {
        let node_index = self.nodes_map.get(node_id).unwrap();
        self.node_values[*node_index]
            .0
            .as_mut()
            .copy_from_slice(values);
    }

    pub fn set_values_from_bool(&mut self, node_id: &NodeId, values: &[bool]) {
        let node_index = self.nodes_map.get(node_id).unwrap();
        let iter = self.node_values[*node_index].0.as_mut().iter_mut();

        for (i, v) in values.iter().zip(iter) {
            *v = if *i { 1. } else { -1. };
        }
    }

    pub fn set_predictions(&mut self, node_id: &NodeId, values: &[f64]) {
        let node_index = self.nodes_map.get(node_id).unwrap();
        self.node_predictions[*node_index]
            .0
            .as_mut()
            .copy_from_slice(values);
    }

    pub fn fix_node(&mut self, node_id: &NodeId, values: &[f64]) {
        self.set_node_type(node_id, NodeType::Fixed);
        self.set_values(node_id, values);
        self.set_predictions(node_id, values);
    }

    pub fn fix_node_from_bool(&mut self, node_id: &NodeId, values: &[bool]) {
        self.set_node_type(node_id, NodeType::Fixed);
        self.set_values_from_bool(node_id, values);
        self.set_predictions_from_bool(node_id, values);
    }

    pub fn set_predictions_from_bool(&mut self, node_id: &NodeId, values: &[bool]) {
        let node_index = self.nodes_map.get(node_id).unwrap();
        let iter = self.node_predictions[*node_index].0.as_mut().iter_mut();

        for (i, v) in values.iter().zip(iter) {
            *v = if *i { 1. } else { -1. };
        }
    }

    pub fn set_node_type(&mut self, node_id: &NodeId, node_type: NodeType) {
        let node_index = self.nodes_map.get(node_id).unwrap();
        self.node_types[*node_index] = node_type;
    }

    pub fn node_values(&self, node_id: &NodeId) -> &[f64] {
        let node_index = self.nodes_map.get(node_id).unwrap();
        self.node_values[*node_index].0.as_ref()
    }
}

type NodeIndex = usize;

struct NodeValues(Box<[f64]>);

impl NodeValues {
    fn new(size: usize) -> Self {
        Self(vec![0.; size].into_boxed_slice())
    }
}

struct NodePredictions(Box<[f64]>);

impl NodePredictions {
    fn new(size: usize) -> Self {
        Self(vec![0.; size].into_boxed_slice())
    }
}

struct NodePredictionDiffs(Box<[f64]>);

impl NodePredictionDiffs {
    fn new(size: usize) -> Self {
        Self(vec![0.; size].into_boxed_slice())
    }
}

struct NodeErrors(Box<[f64]>);

impl NodeErrors {
    fn new(size: usize) -> Self {
        Self(vec![0.; size].into_boxed_slice())
    }
}

struct Edge {
    source: NodeIndex,
    target: NodeIndex,
    weight_matrix: usize,
}

impl Edge {
    fn new(source: NodeIndex, target: NodeIndex, weight_matrix: usize) -> Self {
        Self {
            source,
            target,
            weight_matrix,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Default)]
pub enum NodeType {
    #[default]
    Internal,
    Sensor,
    Fixed,
}

impl NodeType {
    pub fn update_predictions(&self) -> bool {
        *self != Self::Fixed
    }

    pub fn is_sensor(&self) -> bool {
        *self == Self::Sensor
    }
}
