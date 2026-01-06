use crate::activation::ActivationFn;
use crate::dmatrix::DMatrix;
use crate::spec::NodeId;
use petgraph::graph::DefaultIx;
use petgraph::graph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::prelude::EdgeRef;
use petgraph::Direction;
use rand::Rng;
use std::collections::HashMap;

pub(crate) enum PCNode {
    Internal {
        activation_fn: ActivationFn,
        values: Vec<f64>,
        predictions: Vec<f64>,
        errors: Vec<f64>,
    },
    Sensor {
        activation_fn: ActivationFn,
        values: Vec<f64>,
        predictions: Vec<f64>,
        errors: Vec<f64>,
        mask: Vec<bool>,
    },
    Memory {
        activation_fn: ActivationFn,
        values: Vec<f64>,
        errors: Vec<f64>,
    },
}

impl PCNode {
    fn activation_fn(&self) -> &ActivationFn {
        match self {
            PCNode::Internal { activation_fn, .. } => activation_fn,
            PCNode::Sensor { activation_fn, .. } => activation_fn,
            PCNode::Memory { activation_fn, .. } => activation_fn,
        }
    }

    fn values(&self) -> &[f64] {
        match self {
            PCNode::Internal { values, .. } => values,
            PCNode::Sensor { values, .. } => values,
            PCNode::Memory { values, .. } => values,
        }
    }

    pub fn new_internal(size: usize, activation_fn: ActivationFn) -> Self {
        PCNode::Internal {
            activation_fn,
            values: vec![0.; size],
            predictions: vec![0.; size],
            errors: vec![0.; size],
        }
    }

    pub fn new_sensor(size: usize, activation_fn: ActivationFn) -> Self {
        PCNode::Sensor {
            activation_fn,
            values: vec![0.; size],
            predictions: vec![0.; size],
            errors: vec![0.; size],
            mask: vec![false; size],
        }
    }

    pub fn new_memory(size: usize, activation_fn: ActivationFn) -> Self {
        PCNode::Memory {
            activation_fn,
            values: vec![0.; size],
            errors: vec![0.; size],
        }
    }

    pub fn size(&self) -> usize {
        match self {
            PCNode::Internal { values, .. } => values.len(),
            PCNode::Sensor { values, .. } => values.len(),
            PCNode::Memory { values, .. } => values.len(),
        }
    }

    #[allow(dead_code)]
    pub fn energy(&self) -> f64 {
        let mut acc = 0.;

        use PCNode::*;

        let errors = match self {
            Internal { errors, .. } => errors,
            Sensor { errors, .. } => errors,
            Memory { errors, .. } => errors,
        };

        for e in errors {
            acc += e * e;
        }

        0.5 * acc
    }

    pub fn set_predictions(&mut self, p: &[f64]) {
        match self {
            PCNode::Internal { predictions, .. } => predictions.copy_from_slice(p),
            PCNode::Sensor { predictions, .. } => predictions.copy_from_slice(p),
            _ => {}
        }
    }

    pub fn compute_error(&mut self) {
        match self {
            PCNode::Internal {
                predictions,
                errors,
                values,
                ..
            } => {
                for i in 0..errors.len() {
                    errors[i] = values[i] - predictions[i];
                }
            }
            PCNode::Sensor {
                predictions,
                errors,
                values,
                ..
            } => {
                for i in 0..errors.len() {
                    errors[i] = values[i] - predictions[i];
                }
            }
            _ => {}
        }
    }

    pub fn activation(&self, output: &mut [f64]) {
        self.activation_fn().eval(self.values(), output)
    }

    pub fn activation_diff(&self, output: &mut [f64]) {
        self.activation_fn().diff(self.values(), output)
    }

    pub fn activation_diff_mul(&self, output: &mut [f64]) {
        self.activation_fn().diff_mul(self.values(), output)
    }

    #[allow(dead_code)]
    pub fn randomize(&mut self, amount: f64, rng: &mut impl Rng) {
        match self {
            PCNode::Internal { values, .. } => {
                for i in 0..values.len() {
                    values[i] = (1. - amount) * values[i] + amount * rng.random_range(-1. ..1.);
                }
            }
            PCNode::Sensor { values, mask, .. } => {
                for i in 0..values.len() {
                    if !mask[i] {
                        values[i] = (1. - amount) * values[i] + amount * rng.random_range(-1. ..1.);
                    }
                }
            }
            PCNode::Memory { values, .. } => {
                for i in 0..values.len() {
                    values[i] = (1. - amount) * values[i] + amount * rng.random_range(-1. ..1.);
                }
            }
        }
    }
}

pub(crate) struct PCEdge {
    weights: DMatrix<f64>,
}

impl PCEdge {
    pub fn new(from_size: usize, to_size: usize) -> Self {
        Self {
            weights: DMatrix::new(to_size, from_size, 0.),
        }
    }

    #[allow(dead_code)]
    pub fn randomize(&mut self, amount: f64, rng: &mut impl Rng) {
        for r in 0..self.weights.rows() {
            for c in 0..self.weights.cols() {
                self.weights[(r, c)] += amount * rng.random_range(-1. ..1.);
            }
        }
    }
}

type NodeIdx = NodeIndex<DefaultIx>;

pub struct PCN(Graph<PCNode, PCEdge>, HashMap<NodeId, NodeIdx>);

impl PCN {
    pub(crate) fn new(
        graph: Graph<PCNode, PCEdge>,
        nodes_map: HashMap<NodeId, NodeIndex<DefaultIx>>,
    ) -> Self {
        Self(graph, nodes_map)
    }

    fn node_size(&self, node_index: &NodeIdx) -> usize {
        self.0.node_weight(*node_index).unwrap().size()
    }

    pub fn compute_predictions(&mut self) {
        for node_index in self.0.node_indices() {
            let mut node_predictions = vec![0.; self.node_size(&node_index)];

            for edge in self.0.edges_directed(node_index, Direction::Incoming) {
                let n2 = edge.source();
                let mut temp_vec = vec![0.; self.node_size(&n2)];

                self.0.node_weight(n2).unwrap().activation(&mut temp_vec);

                edge.weight()
                    .weights
                    .mul_vec_add(&temp_vec, &mut node_predictions);
            }

            self.0
                .node_weight_mut(node_index)
                .unwrap()
                .set_predictions(&node_predictions);
        }
    }

    pub fn compute_errors(&mut self) {
        for node_index in self.0.node_indices() {
            self.0.node_weight_mut(node_index).unwrap().compute_error();
        }
    }

    pub fn compute_values(&mut self, gamma: f64) {
        for node_index in self.0.node_indices() {
            let w = self.0.node_weight_mut(node_index).unwrap();

            let mut acc = vec![0.; w.size()];

            for edge in self.0.edges_directed(node_index, Direction::Outgoing) {
                let n2 = edge.target();
                edge.weight()
                    .weights
                    .trans_mul_vec_add(self.get_node_errors(&n2), &mut acc);
            }

            self.0
                .node_weight(node_index)
                .unwrap()
                .activation_diff_mul(&mut acc);

            let es = self.get_node_errors(&node_index);
            for i in 0..acc.len() {
                acc[i] -= es[i];
                acc[i] *= gamma;
            }

            self.update_node_values(&node_index, &acc);
        }
    }

    pub fn inference_steps(&mut self, gamma: f64, steps: usize) {
        for _i in 0..steps {
            self.compute_predictions();
            self.compute_errors();
            self.compute_values(gamma);
        }
    }

    pub fn learning_step(&mut self, alpha: f64) {
        for edge_index in self.0.edge_indices() {
            if let Some((source, target)) = self.0.edge_endpoints(edge_index) {
                let errors = self.get_node_errors(&target);
                let values_size = self.node_size(&source);
                let mut temp_values = vec![0.; values_size];
                let mut temp_errors = vec![0.; errors.len()];

                self.0
                    .node_weight(source)
                    .unwrap()
                    .activation_diff(&mut temp_values);

                temp_errors.copy_from_slice(&errors);

                self.0
                    .edge_weight_mut(edge_index)
                    .unwrap()
                    .weights
                    .add_vecs_mul(alpha, &temp_errors, &temp_values);
            }
        }

        for node_index in self.0.node_indices() {
            if let Some(PCNode::Memory { values, errors, .. }) = self.0.node_weight_mut(node_index)
            {
                for i in 0..values.len() {
                    values[i] -= alpha * errors[i];
                }
            }
        }
    }

    pub fn get_node_values(&self, id: NodeId) -> Option<&[f64]> {
        let index = self.1.get(&id)?;
        match self.0.node_weight(*index).unwrap() {
            PCNode::Sensor { values, .. } => Some(values),
            PCNode::Internal { values, .. } => Some(values),
            PCNode::Memory { values, .. } => Some(values),
        }
    }

    fn update_node_values(&mut self, index: &NodeIdx, delta: &[f64]) {
        match self.0.node_weight_mut(*index).unwrap() {
            PCNode::Sensor { values, mask, .. } => {
                for i in 0..values.len() {
                    values[i] += if !mask[i] { delta[i] } else { 0. };
                }
            }
            PCNode::Internal { values, .. } => {
                for i in 0..values.len() {
                    values[i] += delta[i];
                }
            }
            PCNode::Memory { values, .. } => {
                for i in 0..values.len() {
                    values[i] += delta[i];
                }
            }
        }
    }

    fn get_node_errors(&self, index: &NodeIdx) -> &[f64] {
        match self.0.node_weight(*index).unwrap() {
            PCNode::Sensor { errors, .. } => errors,
            PCNode::Internal { errors, .. } => errors,
            PCNode::Memory { errors, .. } => errors,
        }
    }

    #[allow(dead_code)]
    fn get_node_energy(&self, index: &NodeIdx) -> f64 {
        self.0.node_weight(*index).unwrap().energy()
    }

    pub fn set_sensor_values(&mut self, node_id: NodeId, new_values: &[f64], new_mask: &[bool]) {
        let index = self.1.get(&node_id).unwrap();
        let Some(w) = self.0.node_weight_mut(*index) else {
            panic!("Undefined node: {:?}", node_id);
        };

        debug_assert_eq!(w.size(), new_values.len());
        debug_assert_eq!(new_values.len(), new_mask.len());

        match w {
            PCNode::Sensor { mask, values, .. } => {
                for i in 0..values.len() {
                    mask[i] = new_mask[i];

                    if mask[i] {
                        values[i] = new_values[i];
                    }
                }
            }
            _ => panic!("Not a sensor node: {:?}", node_id),
        }
    }

    pub fn set_layer_values(&mut self, node_id: NodeId, new_values: &[f64]) {
        let index = self.1.get(&node_id).unwrap();
        let Some(w) = self.0.node_weight_mut(*index) else {
            panic!("Undefined node: {:?}", node_id);
        };

        debug_assert_eq!(w.size(), new_values.len());

        match w {
            PCNode::Memory { values, .. } => values.copy_from_slice(new_values),
            PCNode::Internal { values, .. } => values.copy_from_slice(new_values),
            PCNode::Sensor { values, mask, .. } => {
                values.copy_from_slice(new_values);
                let l = mask.len();
                mask.copy_from_slice(&vec![false; l]);
            }
        }
    }

    #[allow(dead_code)]
    pub fn randomize_node(&mut self, node_id: NodeId, amount: f64, rng: &mut impl Rng) {
        let index = self.1.get(&node_id).unwrap();
        let Some(w) = self.0.node_weight_mut(*index) else {
            panic!("Undefined node: {:?}", node_id);
        };

        w.randomize(amount, rng);
    }
}
