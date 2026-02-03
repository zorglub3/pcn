use crate::activation::ActivationFn;
use crate::dmatrix::DMatrix;
use crate::dvector::add_inplace;
use crate::dvector::hadamard_inplace;
use crate::spec::NodeId;
use petgraph::graph::DefaultIx;
use petgraph::graph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::prelude::EdgeRef;
use petgraph::Direction;
use rand::Rng;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;

// TODO bugfix - support multiple connections to-/from- nodes
// Check how this affects predictions, value propagation and learning step
//
// TODO refactor to
/*
pub(crate) struct PCNode {
    activation_fn: ActivationFn,
    values: Vec<f64>,
    predictions: Vec<f64>,
    errors: Vec<f64>,
    mask: bool,
}

pub(crate) enum PCNodeKind {
    Internal,
    Sensor,
    Memory,
}
*/

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
        predictions: Vec<f64>,
        memory_pattern: Vec<f64>,
        fix_memory: bool,
    },
}

impl PCNode {
    fn kind_str(&self) -> &'static str {
        match self {
            PCNode::Internal { .. } => "internal",
            PCNode::Sensor { .. } => "sensor",
            PCNode::Memory { .. } => "memory",
        }
    }

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

    fn predictions(&self) -> &[f64] {
        match self {
            PCNode::Internal { predictions, .. } => predictions,
            PCNode::Sensor { predictions, .. } => predictions,
            PCNode::Memory { predictions, .. } => predictions,
        }
    }

    fn errors(&self) -> &[f64] {
        match self {
            PCNode::Internal { errors, .. } => errors,
            PCNode::Sensor { errors, .. } => errors,
            PCNode::Memory { errors, .. } => errors,
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
            predictions: vec![0.; size],
            fix_memory: false,
            memory_pattern: vec![0.; size],
        }
    }

    pub fn size(&self) -> usize {
        match self {
            PCNode::Internal { values, .. } => values.len(),
            PCNode::Sensor { values, .. } => values.len(),
            PCNode::Memory { values, .. } => values.len(),
        }
    }

    // #[allow(dead_code)]
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
            PCNode::Internal { predictions, .. } | PCNode::Sensor { predictions, .. } => {
                predictions.copy_from_slice(p);
            }
            PCNode::Memory {
                predictions,
                fix_memory,
                memory_pattern,
                ..
            } => {
                if *fix_memory {
                    predictions.copy_from_slice(memory_pattern);
                } else {
                    predictions.copy_from_slice(p);
                }
            }
        }
    }

    pub fn compute_error(&mut self) -> f64 {
        let mut err_sum_sqr = 0.;

        match self {
            PCNode::Internal {
                predictions,
                errors,
                values,
                ..
            }
            | PCNode::Sensor {
                predictions,
                errors,
                values,
                ..
            }
            | PCNode::Memory {
                predictions,
                errors,
                values,
                ..
            } => {
                for i in 0..errors.len() {
                    let err = values[i] - predictions[i];
                    err_sum_sqr += err * err;
                    errors[i] = err; // TODO Sigma/variance - that thing
                }
            }
        }

        err_sum_sqr
    }

    /*
    pub fn activation(&self, output: &mut [f64]) {
        self.activation_fn().eval(self.values(), output)
    }
    */

    /*
    pub fn activation_diff_mul(&self, output: &mut [f64]) {
        self.activation_fn().diff_mul(self.values(), output)
    }
    */

    pub fn set_to_random(&mut self, amount: f64, rng: &mut impl Rng) {
        match self {
            PCNode::Internal { values, .. }
            | PCNode::Sensor { values, .. }
            | PCNode::Memory { values, .. } => {
                values.fill_with(|| rng.random_range(-amount..amount));
            }
        }
    }

    pub fn reset(&mut self) {
        use PCNode::*;

        match self {
            Internal {
                values,
                predictions,
                errors,
                ..
            } => {
                values.fill(0.);
                predictions.fill(0.);
                errors.fill(0.);
            }
            Sensor {
                values,
                predictions,
                errors,
                mask,
                ..
            } => {
                values.fill(0.);
                predictions.fill(0.);
                errors.fill(0.);
                mask.fill(false);
            }
            Memory {
                values,
                predictions,
                errors,
                fix_memory,
                ..
            } => {
                values.fill(0.);
                predictions.fill(0.);
                errors.fill(0.);
                *fix_memory = false;
            }
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub enum LearningRule {
    Hebbian,
    Oja,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct WeightMatrix {
    pub matrix: DMatrix<f64>,
    pub learning_rule: LearningRule,
}

#[derive(Clone)]
pub(crate) struct PCEdge {
    weight_matrix_index: usize,
}

impl PCEdge {
    pub fn new(weight_matrix_index: usize) -> Self {
        Self {
            weight_matrix_index,
        }
    }
}

type NodeIdx = NodeIndex<DefaultIx>;

#[allow(clippy::upper_case_acronyms)]
pub struct PCN {
    graph: Graph<PCNode, PCEdge>,
    nodes_map: HashMap<NodeId, NodeIdx>,
    matrices: Vec<WeightMatrix>,
}

impl PCN {
    pub(crate) fn new(
        graph: Graph<PCNode, PCEdge>,
        nodes_map: HashMap<NodeId, NodeIndex<DefaultIx>>,
        matrices: Vec<WeightMatrix>,
    ) -> Self {
        Self {
            graph,
            nodes_map,
            matrices,
        }
    }

    fn node_size(&self, node_index: &NodeIdx) -> usize {
        self.graph.node_weight(*node_index).unwrap().size()
    }

    fn edge_weight_matrix<'a>(&'a self, edge_weight: &PCEdge) -> &'a DMatrix<f64> {
        &self.matrices[edge_weight.weight_matrix_index].matrix
    }

    pub fn compute_predictions(&mut self) {
        for node_index in self.graph.node_indices() {
            let node_size = self.node_size(&node_index);
            let mut node_predictions = vec![0.; node_size];
            let mut acc = vec![0.; node_size];

            for edge in self.graph.edges_directed(node_index, Direction::Incoming) {
                let n2 = edge.source();

                self.edge_weight_matrix(edge.weight())
                    .mul_vec_add(self.graph.node_weight(n2).unwrap().values(), &mut acc);
            }

            self.graph
                .node_weight(node_index)
                .unwrap()
                .activation_fn()
                .eval(&acc, &mut node_predictions);

            self.graph
                .node_weight_mut(node_index)
                .unwrap()
                .set_predictions(&node_predictions);
        }
    }

    pub fn compute_errors(&mut self) -> f64 {
        let mut err_sum_sqr = 0.;
        for node_weight in self.graph.node_weights_mut() {
            err_sum_sqr += node_weight.compute_error();
        }
        err_sum_sqr
    }

    pub fn compute_values(&mut self, gamma: f64) {
        for node_index in self.graph.node_indices() {
            let w = self.graph.node_weight(node_index).unwrap();

            let mut acc = vec![0.; w.size()];

            for edge in self.graph.edges_directed(node_index, Direction::Outgoing) {
                let n2 = edge.target();
                let w2 = self.graph.node_weight(n2).unwrap();
                let mut a = vec![0.; w2.size()];
                self.edge_weight_matrix(edge.weight())
                    .mul_vec(w.values(), &mut a); // a = W x_source
                let mut b = vec![0.; w2.size()];
                w2.activation_fn().diff(&a, &mut b); // b = f'(W x_source)
                hadamard_inplace(w2.errors(), &mut b); // b = f'(W x_source) * e_target
                self.edge_weight_matrix(edge.weight())
                    .trans_mul_vec_add(&b, &mut acc); // acc += W^T (f'(W x_source) * e_target)
            }

            let es = self.node_errors(&node_index);
            for (i, item) in acc.iter_mut().enumerate() {
                *item -= es[i]; // acc = W^T (f'(a) * e_target) - e_source
                *item *= gamma; // acc = gamma (W^T (f'(W x_source) * e_target) - e_source)
            }

            self.update_node_values(&node_index, &acc);
        }
    }

    pub fn inference_step(&mut self, gamma: f64) -> f64 {
        self.compute_predictions();
        let err_sum_sqr = self.compute_errors();
        self.compute_values(gamma);
        err_sum_sqr
    }

    pub fn inference_steps(&mut self, gamma: f64, steps: usize) {
        for _i in 0..steps {
            let _err = self.inference_step(gamma);
            // println!("step {_i}, error={_err}");
            // self.pp();
        }
    }

    pub fn compute_total_energy(&mut self) -> f64 {
        self.compute_predictions();
        self.compute_errors();
        self.get_total_energy()
    }

    pub fn inference_converge(&mut self, gamma: f64, threshold: f64) {
        let mut energy = self.get_total_energy();

        loop {
            self.inference_step(gamma);
            let new_energy = self.get_total_energy();
            if (energy - new_energy) < threshold {
                break;
            }
            energy = new_energy;
        }
    }

    pub fn infer_and_learn(&mut self, gamma: f64, alpha: f64, steps: usize) {
        for _i in 0..steps {
            self.inference_step(gamma);
            self.learning_step(alpha);
        }
    }

    pub fn learning_step(&mut self, alpha: f64) {
        for edge_index in self.graph.edge_indices() {
            let (source, target) = self.graph.edge_endpoints(edge_index).unwrap();
            let source_node = self.graph.node_weight(source).unwrap();
            let target_node = self.graph.node_weight(target).unwrap();
            let matrix_index = self
                .graph
                .edge_weight(edge_index)
                .unwrap()
                .weight_matrix_index;

            let mut a = vec![0.; target_node.size()];

            self.matrices[matrix_index]
                .matrix
                .mul_vec(source_node.values(), &mut a);
            // a = W x_source

            let mut b = vec![0.; target_node.size()];
            target_node.activation_fn().diff(&a, &mut b); // b = f'(W x_source)
            hadamard_inplace(target_node.errors(), &mut b); // b = f'(W x_source) * e_target

            let matrix = &mut self.matrices[matrix_index].matrix;
            let values = &source_node.values();

            // TODO select learning rule by matrix
            for r in 0..matrix.rows() {
                for c in 0..matrix.cols() {
                    // TODO - find the right version of Oja's rule to use.
                    // line below or the other one
                    // let delta = alpha * b[r] * (values[c] - b[r] * matrix[(r, c)]);
                    // delta = alpha * (f'(W x_source) * e_target) (x_source - ...)
                    let delta = alpha * values[c] * (b[r] - values[c] * matrix[(r, c)]);
                    matrix[(r, c)] += delta;
                }
            }

            // self.matrices[matrix_index].add_vecs_mul(alpha, &b, &source_node.values());
        }
    }

    pub fn get_node_values(&self, id: NodeId) -> &[f64] {
        let index = self.nodes_map.get(&id).unwrap();
        match self.graph.node_weight(*index).unwrap() {
            PCNode::Sensor { values, .. } => values,
            PCNode::Internal { values, .. } => values,
            PCNode::Memory { values, .. } => values,
        }
    }

    pub fn get_node_predictions(&self, id: NodeId) -> &[f64] {
        let index = self.nodes_map.get(&id).unwrap();
        match self.graph.node_weight(*index).unwrap() {
            PCNode::Sensor { predictions, .. } => predictions,
            PCNode::Internal { predictions, .. } => predictions,
            PCNode::Memory { predictions, .. } => predictions,
        }
    }

    fn update_node_values(&mut self, index: &NodeIdx, delta: &[f64]) {
        match self.graph.node_weight_mut(*index).unwrap() {
            PCNode::Sensor { values, mask, .. } => {
                for i in 0..values.len() {
                    if !mask[i] {
                        values[i] += delta[i];
                    }
                }
            }
            PCNode::Internal { values, .. } => add_inplace(delta, values),
            PCNode::Memory { values, .. } => add_inplace(delta, values),
        }
    }

    fn node_errors(&self, index: &NodeIdx) -> &[f64] {
        match self.graph.node_weight(*index).unwrap() {
            PCNode::Sensor { errors, .. } => errors,
            PCNode::Internal { errors, .. } => errors,
            PCNode::Memory { errors, .. } => errors,
        }
    }

    pub fn get_node_energy(&self, node_id: NodeId) -> f64 {
        let index = self.nodes_map.get(&node_id).unwrap();
        self.graph.node_weight(*index).unwrap().energy()
    }

    pub fn get_total_energy(&self) -> f64 {
        self.graph
            .node_weights()
            .map(|node_weight| node_weight.energy())
            .sum()
    }

    pub fn set_memory_values(&mut self, node_id: NodeId, values: &[f64]) {
        let index = self.nodes_map.get(&node_id).unwrap();
        let w = self.graph.node_weight_mut(*index).unwrap();

        debug_assert_eq!(w.size(), values.len());

        match w {
            PCNode::Memory {
                fix_memory,
                memory_pattern,
                ..
            } => {
                memory_pattern.copy_from_slice(values);
                *fix_memory = true;
            }
            _ => panic!("Not a memory node: {:?}", node_id),
        }
    }

    pub fn unset_memory(&mut self, node_id: NodeId) {
        let index = self.nodes_map.get(&node_id).unwrap();
        let w = self.graph.node_weight_mut(*index).unwrap();

        match w {
            PCNode::Memory { fix_memory, .. } => {
                *fix_memory = true;
            }
            _ => panic!("Not a memory node: {:?}", node_id),
        }
    }

    pub fn set_sensor_values(&mut self, node_id: NodeId, new_values: &[f64], new_mask: &[bool]) {
        let index = self.nodes_map.get(&node_id).unwrap();
        let Some(w) = self.graph.node_weight_mut(*index) else {
            panic!("Undefined node: {:?}", node_id);
        };

        debug_assert_eq!(w.size(), new_values.len());
        debug_assert_eq!(new_values.len(), new_mask.len());

        match w {
            PCNode::Sensor { mask, values, .. } => {
                mask.copy_from_slice(new_mask);
                values.copy_from_slice(new_values);
            }
            _ => panic!("Not a sensor node: {:?}", node_id),
        }
    }

    pub fn set_layer_values(&mut self, node_id: NodeId, new_values: &[f64]) {
        let index = self.nodes_map.get(&node_id).unwrap();
        let Some(w) = self.graph.node_weight_mut(*index) else {
            panic!("Undefined node: {:?}", node_id);
        };

        debug_assert_eq!(w.size(), new_values.len());

        match w {
            PCNode::Memory { values, .. } => values.copy_from_slice(new_values),
            PCNode::Internal { values, .. } => values.copy_from_slice(new_values),
            PCNode::Sensor { values, .. } => values.copy_from_slice(new_values),
        }
    }

    pub fn randomize_node(&mut self, node_id: NodeId, amount: f64, rng: &mut impl Rng) {
        let index = self.nodes_map.get(&node_id).unwrap();
        let Some(w) = self.graph.node_weight_mut(*index) else {
            panic!("Undefined node: {:?}", node_id);
        };

        w.set_to_random(amount, rng);
    }

    pub fn randomize_all_nodes(&mut self, amount: f64, rng: &mut impl Rng) {
        for node_weight in self.graph.node_weights_mut() {
            node_weight.set_to_random(amount, rng);
        }
    }

    pub fn reset_node(&mut self, node_id: NodeId) {
        let index = self.nodes_map.get(&node_id).unwrap();
        self.graph.node_weight_mut(*index).unwrap().reset();
    }

    pub fn reset_all_nodes(&mut self) {
        for node_weight in self.graph.node_weights_mut() {
            node_weight.reset();
        }
    }

    pub fn pp(&self) {
        println!("# Nodes");
        for node_id in self.nodes_map.keys() {
            let node_index = self.nodes_map.get(node_id).unwrap();
            let node_data = self.graph.node_weight(*node_index).unwrap();
            println!("- Node {:?}:", &node_id);
            println!("  + type  : {}", node_data.kind_str());
            println!("  + values: {:?}", node_data.values());
            println!("  + errors: {:?}", node_data.errors());
            println!("  + predictions: {:?}", node_data.predictions());
            println!("  + energy: {:?}", node_data.energy());
        }
        println!();

        /*
        println!("# Edges");
        for edge_weight in self.graph.edge_weights() {
            println!("- Edge:");
            println!("  + weight matrix: {:?}", edge_weight.weight_matrix_index);
        }
        println!();

        println!("# Matrices");
        for (n, matrix) in self.matrices.iter().enumerate() {
            println!("Matrix {}:", n);
            matrix.matrix.pp();
        }
        println!();
        */
    }
}
