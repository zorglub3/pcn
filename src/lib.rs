mod activation;
mod dmatrix;

pub use activation::ActivationFn;
use dmatrix::DMatrix;
use petgraph::graph::DefaultIx;
use petgraph::graph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::prelude::EdgeRef;
use petgraph::Direction;
use rand::Rng;

#[derive(Debug)]
pub enum PCError {
    EdgeFromSensor,
    UndefinedNode(NodeId),
    NotSensor(NodeId),
    NotMemory(NodeId),
}

enum PCNode {
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
        match self {
            PCNode::Internal {
                values,
                activation_fn,
                ..
            } => activation_fn.eval(values, output),
            PCNode::Sensor {
                values,
                activation_fn,
                ..
            } => activation_fn.eval(values, output),
            PCNode::Memory {
                values,
                activation_fn,
                ..
            } => activation_fn.eval(values, output),
        }
    }

    pub fn activation_diff(&self, output: &mut [f64]) {
        match self {
            PCNode::Internal {
                values,
                activation_fn,
                ..
            } => activation_fn.diff(values, output),
            PCNode::Sensor {
                values,
                activation_fn,
                ..
            } => activation_fn.diff(values, output),
            PCNode::Memory {
                values,
                activation_fn,
                ..
            } => activation_fn.diff(values, output),
        }
    }

    pub fn activation_diff_mul(&self, output: &mut [f64]) {
        match self {
            PCNode::Internal {
                values,
                activation_fn,
                ..
            } => activation_fn.diff_mul(values, output),
            PCNode::Sensor {
                values,
                activation_fn,
                ..
            } => activation_fn.diff_mul(values, output),
            PCNode::Memory {
                values,
                activation_fn,
                ..
            } => activation_fn.diff_mul(values, output),
        }
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

struct PCEdge {
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

pub struct PCN(Graph<PCNode, PCEdge>);

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub struct NodeId(NodeIndex<DefaultIx>);

impl PCN {
    pub fn new() -> Self {
        Self(Graph::<PCNode, PCEdge>::new())
    }

    pub fn is_sensor(&self, node_id: &NodeId) -> bool {
        match self.0.node_weight(node_id.0) {
            Some(PCNode::Sensor { .. }) => true,
            _ => false,
        }
    }

    pub fn is_internal(&self, node_id: &NodeId) -> bool {
        match self.0.node_weight(node_id.0) {
            Some(PCNode::Internal { .. }) => true,
            _ => false,
        }
    }

    pub fn node_size(&self, node_id: &NodeId) -> Result<usize, PCError> {
        let Some(w) = self.0.node_weight(node_id.0) else {
            return Err(PCError::UndefinedNode(*node_id));
        };

        Ok(w.size())
    }

    pub fn add_internal_node(&mut self, node_size: usize, activation_fn: ActivationFn) -> NodeId {
        NodeId(
            self.0
                .add_node(PCNode::new_internal(node_size, activation_fn)),
        )
    }

    pub fn add_sensor_node(&mut self, node_size: usize, activation_fn: ActivationFn) -> NodeId {
        NodeId(
            self.0
                .add_node(PCNode::new_sensor(node_size, activation_fn)),
        )
    }

    pub fn add_memory_node(&mut self, node_size: usize, activation_fn: ActivationFn) -> NodeId {
        NodeId(
            self.0
                .add_node(PCNode::new_memory(node_size, activation_fn)),
        )
    }

    pub fn connect_nodes(&mut self, source: &NodeId, target: &NodeId) -> Result<(), PCError> {
        if self.is_sensor(source) {
            Err(PCError::EdgeFromSensor)
        } else {
            match (self.0.node_weight(source.0), self.0.node_weight(target.0)) {
                (Some(n1w), Some(n2w)) => {
                    self.0
                        .add_edge(source.0, target.0, PCEdge::new(n1w.size(), n2w.size()));
                    Ok(())
                }
                (None, _) => Err(PCError::UndefinedNode(*source)),
                (_, None) => Err(PCError::UndefinedNode(*target)),
            }
        }
    }

    pub fn compute_predictions(&mut self) -> Result<(), PCError> {
        for node_index in self.0.node_indices() {
            let mut node_predictions = vec![0.; self.node_size(&NodeId(node_index))?];

            for edge in self.0.edges_directed(node_index, Direction::Incoming) {
                let n2 = edge.source();
                let mut temp_vec = vec![0.; self.node_size(&NodeId(n2))?];

                self.0
                    .node_weight(n2)
                    .ok_or(PCError::UndefinedNode(NodeId(n2)))?
                    .activation(&mut temp_vec);

                edge.weight()
                    .weights
                    .mul_vec_add(&temp_vec, &mut node_predictions);
            }

            let Some(w) = self.0.node_weight_mut(node_index) else {
                return Err(PCError::UndefinedNode(NodeId(node_index)));
            };

            w.set_predictions(&node_predictions);
        }

        Ok(())
    }

    pub fn compute_errors(&mut self) -> Result<(), PCError> {
        for node_index in self.0.node_indices() {
            let Some(w) = self.0.node_weight_mut(node_index) else {
                return Err(PCError::UndefinedNode(NodeId(node_index)));
            };

            w.compute_error();
        }

        Ok(())
    }

    pub fn compute_values(&mut self, gamma: f64) -> Result<(), PCError> {
        for node_index in self.0.node_indices() {
            let Some(w) = self.0.node_weight_mut(node_index) else {
                return Err(PCError::UndefinedNode(NodeId(node_index)));
            };

            let mut acc = vec![0.; w.size()];

            for edge in self.0.edges_directed(node_index, Direction::Outgoing) {
                let n2 = edge.target();
                edge.weight()
                    .weights
                    .trans_mul_vec_add(self.get_node_errors(&NodeId(n2))?, &mut acc);
            }

            self.0
                .node_weight(node_index)
                .ok_or(PCError::UndefinedNode(NodeId(node_index)))?
                .activation_diff_mul(&mut acc);

            let es = self.get_node_errors(&NodeId(node_index))?;
            for i in 0..acc.len() {
                acc[i] -= es[i];
                acc[i] *= gamma;
            }

            self.update_node_values(&NodeId(node_index), &acc)?;
        }

        Ok(())
    }

    pub fn inference_steps(&mut self, gamma: f64, steps: usize) -> Result<(), PCError> {
        for _i in 0..steps {
            self.compute_predictions()?;
            self.compute_errors()?;
            self.compute_values(gamma)?;
        }

        Ok(())
    }

    pub fn learning_step(&mut self, alpha: f64) -> Result<(), PCError> {
        for edge_index in self.0.edge_indices() {
            if let Some((source, target)) = self.0.edge_endpoints(edge_index) {
                let errors = self.get_node_errors(&NodeId(target))?;
                let values_size = self.node_size(&NodeId(source))?;
                let mut temp_values = vec![0.; values_size];
                let mut temp_errors = vec![0.; errors.len()];

                self.0
                    .node_weight(source)
                    .ok_or(PCError::UndefinedNode(NodeId(source)))?
                    .activation_diff(&mut temp_values);
                temp_errors.copy_from_slice(&errors);

                if let Some(w) = self.0.edge_weight_mut(edge_index) {
                    w.weights.add_vecs_mul(alpha, &temp_errors, &temp_values);
                }
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

        Ok(())
    }

    pub fn get_node_values(&self, node_id: &NodeId) -> Result<&[f64], PCError> {
        match self.0.node_weight(node_id.0) {
            Some(PCNode::Sensor { values, .. }) => Ok(values),
            Some(PCNode::Internal { values, .. }) => Ok(values),
            Some(PCNode::Memory { values, .. }) => Ok(values),
            None => Err(PCError::UndefinedNode(*node_id)),
        }
    }

    pub fn update_node_values(&mut self, node_id: &NodeId, delta: &[f64]) -> Result<(), PCError> {
        match self.0.node_weight_mut(node_id.0) {
            Some(PCNode::Sensor { values, mask, .. }) => {
                for i in 0..values.len() {
                    values[i] += if !mask[i] { delta[i] } else { 0. };
                }
                Ok(())
            }
            Some(PCNode::Internal { values, .. }) => {
                for i in 0..values.len() {
                    values[i] += delta[i];
                }
                Ok(())
            }
            Some(PCNode::Memory { values, .. }) => {
                for i in 0..values.len() {
                    values[i] += delta[i];
                }
                Ok(())
            }
            None => Err(PCError::UndefinedNode(*node_id)),
        }
    }

    pub fn get_node_errors(&self, node_id: &NodeId) -> Result<&[f64], PCError> {
        match self.0.node_weight(node_id.0) {
            Some(PCNode::Sensor { errors, .. }) => Ok(errors),
            Some(PCNode::Internal { errors, .. }) => Ok(errors),
            Some(PCNode::Memory { errors, .. }) => Ok(errors),
            None => Err(PCError::UndefinedNode(*node_id)),
        }
    }

    pub fn get_node_energy(&self, node_id: &NodeId) -> Result<f64, PCError> {
        match self.0.node_weight(node_id.0) {
            Some(node) => Ok(node.energy()),
            None => Err(PCError::UndefinedNode(*node_id)),
        }
    }

    pub fn set_sensor_values(
        &mut self,
        node_id: &NodeId,
        new_values: &[f64],
        new_mask: &[bool],
    ) -> Result<(), PCError> {
        let Some(w) = self.0.node_weight_mut(node_id.0) else {
            return Err(PCError::UndefinedNode(*node_id));
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

                Ok(())
            }
            _ => Err(PCError::NotSensor(*node_id)),
        }
    }

    pub fn set_layer_values(
        &mut self,
        node_id: &NodeId,
        new_values: &[f64],
    ) -> Result<(), PCError> {
        let Some(w) = self.0.node_weight_mut(node_id.0) else {
            return Err(PCError::UndefinedNode(*node_id));
        };

        debug_assert_eq!(w.size(), new_values.len());

        match w {
            PCNode::Memory { values, .. } => Ok(values.copy_from_slice(new_values)),
            PCNode::Internal { values, .. } => Ok(values.copy_from_slice(new_values)),
            PCNode::Sensor { values, mask, .. } => {
                values.copy_from_slice(new_values);
                let l = mask.len();
                mask.copy_from_slice(&vec![false; l]);
                Ok(())
            }
        }
    }

    #[allow(dead_code)]
    pub fn randomize_node(
        &mut self,
        node_id: &NodeId,
        amount: f64,
        rng: &mut impl Rng,
    ) -> Result<(), PCError> {
        let Some(w) = self.0.node_weight_mut(node_id.0) else {
            return Err(PCError::UndefinedNode(*node_id));
        };

        w.randomize(amount, rng);

        Ok(())
    }
}
