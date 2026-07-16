use crate::activation::ActivationFn;
use crate::dmatrix::DMatrix;
use crate::dvector::randomize_vec;
use crate::dvector::scale_add_inplace;
use crate::dvector::sub_inplace;
use rand::Rng;
use std::collections::BTreeMap;

pub struct PCN<NodeId: Eq + Ord + Clone> {
    activation_functions: Vec<ActivationFn>,
    node_values: Vec<NodeValues>,
    node_value_updates: Vec<NodeValueUpdate>,
    node_predictions: Vec<NodePredictions>,
    node_prediction_diffs: Vec<NodePredictionDiffs>,
    node_errors: Vec<NodeErrors>,
    node_sizes: Vec<usize>,
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
            node_value_updates: Vec::new(),
            node_predictions: Vec::new(),
            node_prediction_diffs: Vec::new(),
            node_errors: Vec::new(),
            node_sizes: Vec::new(),
            next_node_index: 0,
            weight_matrices: Vec::new(),
            edges: Vec::new(),
            nodes_map: BTreeMap::new(),
        }
    }
}

impl<NodeId: Eq + Ord + Clone> PCN<NodeId> {
    pub fn add_node(
        &mut self, 
        id: &NodeId,
        activation_function: ActivationFn, 
        size: usize,
    ) {
        debug_assert!(!self.nodes_map.contains_key(&id));

        let node_index = self.next_node_index;
        self.next_node_index += 1;

        self.activation_functions.push(activation_function);
        self.node_values.push(NodeValues::new(size));
        self.node_value_updates.push(NodeValueUpdate::new(size));
        self.node_predictions.push(NodePredictions::new(size));
        self.node_errors.push(NodeErrors::new(size));
        self.node_sizes.push(size);

        self.nodes_map.insert(id.clone(), node_index);
    }

    pub fn add_edge(
        &mut self, 
        target_id: &NodeId,
        source_id: &NodeId, 
    ) {
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
            .push(Edge::new(*source, *target, weight_matrix_index, target_size));
    }

    pub fn randomize_weights<R: Rng>(&mut self, rng: &mut R) {
        for weight_matrix in self.weight_matrices.iter_mut() {
            weight_matrix.randomize(1., rng);
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
            .zip(&self.node_predictions);

        for ((error, value), prediction) in iter {
            let inner_iter = error
                .0
                .iter_mut()
                .zip(value.0.as_ref())
                .zip(prediction.0.as_ref());

            for ((e, v), p) in inner_iter {
                let err = v - p;
                *e = err;
                error_square_sum += err * err;
            }
        }

        error_square_sum
    }

    pub fn compute_predictions(&mut self) {
        for prediction in self.node_predictions.iter_mut() {
            prediction.0.as_mut().fill(0.);
        }

        for edge in self.edges.iter() {
            let matrix = &self.weight_matrices[edge.weight_matrix];
            let source = self.node_values[edge.source].0.as_ref();
            let target = self.node_predictions[edge.target].0.as_mut();

            matrix.mul_vec_add(source, target);
        }

        for (i, prediction) in self.node_predictions.iter_mut().enumerate() {
            self.activation_functions[i].eval_inplace(prediction.0.as_mut());
        }
    }

    pub fn compute_prediction_diffs(&mut self) {
        for prediction_diff in self.node_prediction_diffs.iter_mut() {
            prediction_diff.0.as_mut().fill(0.);
        }

        for edge in self.edges.iter() {
            let matrix = &self.weight_matrices[edge.weight_matrix];
            let source = self.node_values[edge.source].0.as_ref();
            let target = self.node_prediction_diffs[edge.target].0.as_mut();

            matrix.mul_vec_add(source, target);
        }

        for (i, prediction_diff) in self.node_predictions.iter_mut().enumerate() {
            self.activation_functions[i].diff_inplace(prediction_diff.0.as_mut());
        }
    }

    pub fn compute_values(&mut self, gamma: f64) {
        self.compute_prediction_diffs();

        for u in self.node_value_updates.iter_mut() {
            u.0.fill(0.);
        }

        for edge in self.edges.iter() {
            let w = &self.weight_matrices[edge.weight_matrix];
            let pd = &self.node_prediction_diffs[edge.target];
            let e = &self.node_errors[edge.target];

            // TODO compute w^T (pd * e) and add it to node_value_updates
            //  Coult be that last part (adding it to nvu) is done in sep step
        }

        for (e, u) in self.node_errors.iter().zip(self.node_value_updates.iter_mut()) {
            sub_inplace(&e.0, &mut u.0);
        }

        for (v, u) in self.node_values.iter_mut().zip(self.node_value_updates.iter()) {
            scale_add_inplace(gamma, &u.0, &mut v.0);
        }
    }

    pub fn learn_hebb(&mut self, alpha: f64) {
        self.compute_prediction_diffs();

        for edge in self.edges.iter() {
            let w = &mut self.weight_matrices[edge.weight_matrix];
            let h = &self.node_prediction_diffs[edge.target].0.as_ref();
            let e = &self.node_errors[edge.target].0.as_ref();
            let x = &self.node_values[edge.source].0.as_ref();

            for r in w.rows_range() {
                for c in w.cols_range() {
                    w[(r, c)] += alpha * h[r] * e[r] * x[c];
                }
            }
        }
    }

    pub fn learn_oja(&mut self, alpha: f64) {
        todo!()
    }
}

type NodeIndex = usize;
type EdgeIndex = usize;

struct NodeValueUpdate(Box<[f64]>);

impl NodeValueUpdate {
    fn new(size: usize) -> Self {
        Self(vec![0.; size].into_boxed_slice())
    }
}

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
    gain_modulated_error: Box<[f64]>,
}

impl Edge {
    fn new(source: NodeIndex, target: NodeIndex, weight_matrix: usize, target_size: usize) -> Self {
        Self {
            source,
            target,
            weight_matrix,
            gain_modulated_error: vec![0.; target_size].into_boxed_slice(),
        }
    }
}
