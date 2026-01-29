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
        predictions: Vec<f64>,
        memory_pattern: Vec<f64>,
        fix_memory: bool,
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
            PCNode::Internal { predictions, .. } |
            PCNode::Sensor { predictions, .. } => {
                predictions.copy_from_slice(p);
            }
            PCNode::Memory { predictions, fix_memory, memory_pattern, .. } => {
                if *fix_memory {
                    predictions.copy_from_slice(memory_pattern);
                } else {
                    predictions.copy_from_slice(p);
                }
            }
        }
    }

    pub fn compute_error(&mut self) {
        match self {
            PCNode::Internal {
                predictions,
                errors,
                values,
                ..
            } |
            PCNode::Sensor {
                predictions,
                errors,
                values,
                ..
            } |
            PCNode::Memory {
                predictions,
                errors,
                values,
                ..
            } => {
                for i in 0..errors.len() {
                    errors[i] = (values[i] - predictions[i]) / 1.; // TODO Sigma/variance - that thing
                }
            }
        }
    }

    pub fn activation(&self, output: &mut [f64]) {
        self.activation_fn().eval(self.values(), output)
    }

    pub fn activation_diff_mul(&self, output: &mut [f64]) {
        self.activation_fn().diff_mul(self.values(), output)
    }

    pub fn set_to_random(&mut self, amount: f64, rng: &mut impl Rng) {
        match self {
            PCNode::Internal { values, .. } | PCNode::Sensor { values, .. } | PCNode::Memory { values, .. } => {
                values.fill_with(|| rng.random_range(-amount..amount));
            }
        }
    }

    pub fn reset(&mut self) {
        use PCNode::*;

        match self {
            Internal { values, predictions, errors, .. } => {
                values.fill(0.);
                predictions.fill(0.);
                errors.fill(0.);
            }
            Sensor { values, predictions, errors, mask, .. } => {
                values.fill(0.);
                predictions.fill(0.);
                errors.fill(0.);
                mask.fill(false);
            }
            Memory { values, predictions, errors, fix_memory, .. } => {
                values.fill(0.);
                predictions.fill(0.);
                errors.fill(0.);
                *fix_memory = false;
            }
        }
    }
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
    matrices: Vec<DMatrix<f64>>,
}

impl PCN {
    pub(crate) fn new(
        graph: Graph<PCNode, PCEdge>,
        nodes_map: HashMap<NodeId, NodeIndex<DefaultIx>>,
        matrices: Vec<DMatrix<f64>>,
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
        &self.matrices[edge_weight.weight_matrix_index]
    }

    pub fn compute_predictions(&mut self) {
        for node_index in self.graph.node_indices() {
            let mut node_predictions = vec![0.; self.node_size(&node_index)];

            for edge in self.graph.edges_directed(node_index, Direction::Incoming) {
                let n2 = edge.source();
                let mut temp_vec = vec![0.; self.node_size(&n2)];

                self.graph
                    .node_weight(n2)
                    .unwrap()
                    .activation(&mut temp_vec);

                self.edge_weight_matrix(edge.weight())
                    .mul_vec_add(&temp_vec, &mut node_predictions);
            }

            self.graph
                .node_weight_mut(node_index)
                .unwrap()
                .set_predictions(&node_predictions);
        }
    }

    pub fn compute_errors(&mut self) {
        for node_weight in self.graph.node_weights_mut() {
            node_weight.compute_error();
        }
    }

    pub fn compute_values(&mut self, gamma: f64) {
        for node_index in self.graph.node_indices() {
            let w = self.graph.node_weight_mut(node_index).unwrap();

            let mut acc = vec![0.; w.size()];

            for edge in self.graph.edges_directed(node_index, Direction::Outgoing) {
                let n2 = edge.target();
                self.edge_weight_matrix(edge.weight())
                    .trans_mul_vec_add(self.node_errors(&n2), &mut acc);
            }

            self.graph
                .node_weight(node_index)
                .unwrap()
                .activation_diff_mul(&mut acc);

            let es = self.node_errors(&node_index);
            for i in 0..acc.len() {
                acc[i] -= es[i];
                acc[i] *= gamma;
            }

            self.update_node_values(&node_index, &acc);
        }
    }

    pub fn inference_step(&mut self, gamma: f64) {
        self.compute_predictions();
        self.compute_errors();
        self.compute_values(gamma);
        // println!("inference step, energy: {}", self.get_total_energy());
    }

    pub fn inference_steps(&mut self, gamma: f64, steps: usize) {
        // println!("Doing {} inference steps", steps);
        for _i in 0..steps {
            self.inference_step(gamma);
        }
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

            let errors = self.node_errors(&target);
            let values_size = self.node_size(&source);
            let mut temp_values = vec![0.; values_size];
            let mut temp_errors = vec![0.; errors.len()];

            self.graph
                .node_weight(source)
                .unwrap()
                .activation(&mut temp_values);

            temp_errors.copy_from_slice(errors);

            let matrix_index = self
                .graph
                .edge_weight(edge_index)
                .unwrap()
                .weight_matrix_index;

            let matrix = &mut self.matrices[matrix_index];

            for r in 0..matrix.rows() {
                for c in 0..matrix.cols() {
                    // using Oja's rule w_i,j += alpha * y_i * (x_j - /Sum_k=1 w_k,j * y_k)
                    let s = matrix.mul_col_vec(&temp_errors, c);
                    let delta = alpha * temp_errors[r] * (temp_values[c] - s);
                    matrix[(r, c)] += delta;

                    // from PCN papers
                    // let delta = alpha * temp_errors[r] * temp_values[c];
                    // matrix[(r, c)] += delta;
                }
            }
            // self.matrices[matrix_index].add_vecs_mul(alpha, &temp_errors, &temp_values);
        }

        for node_index in self.graph.node_indices() {
            if let Some(PCNode::Memory { values, errors, .. }) =
                self.graph.node_weight_mut(node_index)
            {
                for i in 0..values.len() {
                    values[i] -= alpha * errors[i];
                }
            }
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

    fn update_node_values(&mut self, index: &NodeIdx, delta: &[f64]) {
        match self.graph.node_weight_mut(*index).unwrap() {
            PCNode::Sensor { values, mask, .. } => {
                for i in 0..values.len() {
                    if !mask[i] {
                        values[i] += delta[i];
                    }
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

    /*
    fn node_values(&self, index: &NodeIdx) -> &[f64] {
        match self.graph.node_weight(*index).unwrap() {
            PCNode::Sensor { values, .. } => values,
            PCNode::Internal { values, .. } => values,
            PCNode::Memory { values, .. } => values,
        }
    }
    */

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

    /*
    pub fn set_sensor_predictions(&mut self, node_id: NodeId, new_values: &[f64], new_mask: &[bool]) {
        let index = self.nodes_map.get(&node_id).unwrap();
        let Some(w) = self.graph.node_weight_mut(*index) else {
            panic!("Undefined node: {:?}", node_id);
        };

        debug_assert_eq!(w.size(), new_values.len());
        debug_assert_eq!(new_values.len(), new_mask.len());

        match w {
            PCNode::Sensor { mask, predictions, .. } => {
                mask.copy_from_slice(new_mask);
                predictions.copy_from_slice(new_values);
            }
            _ => panic!("Node {:?} is not a sensor node", node_id),
        }
    }
    */

    pub fn set_memory_values(&mut self, node_id: NodeId, values: &[f64]) {
        let index = self.nodes_map.get(&node_id).unwrap();
        let w = self.graph.node_weight_mut(*index).unwrap();

        debug_assert_eq!(w.size(), values.len());

        match w {
            PCNode::Memory { fix_memory, memory_pattern, .. } => {
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
                for i in 0..values.len() {
                    mask[i] = new_mask[i];

                    values[i] = new_values[i];
                }
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

    // #[allow(dead_code)]
    pub fn randomize_node(&mut self, node_id: NodeId, amount: f64, rng: &mut impl Rng) {
        let index = self.nodes_map.get(&node_id).unwrap();
        let Some(w) = self.graph.node_weight_mut(*index) else {
            panic!("Undefined node: {:?}", node_id);
        };

        w.set_to_random(amount, rng);
    }

    // #[allow(dead_code)]
    pub fn reset_node(&mut self, node_id: NodeId) {
        let index = self.nodes_map.get(&node_id).unwrap();
        self.graph.node_weight_mut(*index).unwrap().reset();
    }

    pub fn reset_all_nodes(&mut self) {
        for node_weight in self.graph.node_weights_mut() {
            node_weight.reset();
        }
    }

    // #[allow(dead_code)]
    pub fn pp(&self) {
        println!("# Nodes");
        for node_id in self.nodes_map.keys() {
            let node_index = self.nodes_map.get(node_id).unwrap();
            let node_data = self.graph.node_weight(*node_index).unwrap();
            println!("- Node {:?}:", &node_id);
            println!("  + values: {:?}", node_data.values());
            println!("  + errors: {:?}", node_data.errors());
            println!("  + energy: {:?}", node_data.energy());
        }
        println!();

        println!("# Edges");
        for edge_weight in self.graph.edge_weights() {
            println!("- Edge:");
            println!("  + weight matrix: {:?}", edge_weight.weight_matrix_index);
        }
        println!();

        println!("# Matrices");
        for (n, matrix) in self.matrices.iter().enumerate() {
            println!("Matrix {}:", n);
            matrix.pp();
        }
        println!();
    }
}
