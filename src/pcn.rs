//! Implementation fo Predictive Coding Network. Loosely based on
//! "Introduction to Predictive Coding Networks for Machine Learning"
//! by Mikko Stenlund.

use crate::activation::ActivationFn;
use dense_matrix::RowWise;
use dense_matrix::Symmetric;
use dense_matrix::Matrix;
use dense_matrix::RowVector;
use dense_matrix::ColumnVector;
use dense_matrix::add_scaled;
use std::collections::BTreeMap;
use std::fmt::{Debug, Error as FmtError, Formatter};
use std::iter::Sum;
use std::ops::{AddAssign, Mul, MulAssign, Sub, SubAssign};

pub struct PCN<N: MulAssign + Mul<Output = N> + Default + AddAssign + SubAssign + Sum + Copy, A: ActivationFn<N>, NodeId: Eq + Ord + Clone> {
    activation_functions: Vec<A>,
    node_values: Vec<NodeValues<N>>,
    node_predictions: Vec<NodePredictions<N>>,
    node_gain_modulated_errors: Vec<GainModulatedErrors<N>>,
    node_errors: Vec<NodeErrors<N>>,
    node_sizes: Vec<usize>,
    node_types: Vec<NodeType>,
    node_in_degree: Vec<usize>,
    node_out_degree: Vec<usize>,
    next_node_index: usize,
    weight_matrices: Vec<WeightMatrix<N>>,
    edges: Vec<Edge>,
    nodes_map: BTreeMap<NodeId, NodeIndex>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct NodeData<'a, N: MulAssign + Mul<Output = N> + Default + AddAssign + Sum + Copy + Debug> {
    values: &'a NodeValues<N>,
    predictions: &'a NodePredictions<N>,
    errors: &'a NodeErrors<N>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct EdgeData<'a, NodeId: Debug> {
    source: &'a NodeId,
    target: &'a NodeId,
    matrix: usize,
}

impl<N: Sub<Output = N> + MulAssign + Mul<Output = N> + Default + AddAssign + Sum + Copy + Debug + SubAssign, A: ActivationFn<N>, NodeId: Eq + Ord + Clone + Debug> Debug for PCN<N, A, NodeId> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_map()
            .entries(self.nodes_map.iter().map(|(k, v)| {
                (
                    k,
                    NodeData {
                        values: &self.node_values[*v],
                        predictions: &self.node_predictions[*v],
                        errors: &self.node_errors[*v],
                    },
                )
            }))
            .finish()?;

        f.debug_list()
            .entries(self.edges.iter().map(|e| EdgeData {
                source: self.reverse_lookup_node(e.source),
                target: self.reverse_lookup_node(e.target),
                matrix: e.weight_matrix_index,
            }))
            .finish()?;

        f.debug_list()
            .entries(self.weight_matrices.iter())
            .finish()?;

        Ok(())
    }
}

impl<N: MulAssign + Mul<Output = N> + Default + AddAssign + Sum + Copy + Debug + SubAssign, A: ActivationFn<N>, NodeId: Eq + Ord + Clone> Default for PCN<N, A, NodeId> {
    fn default() -> Self {
        Self {
            activation_functions: Vec::new(),
            node_values: Vec::new(),
            node_predictions: Vec::new(),
            node_gain_modulated_errors: Vec::new(),
            node_errors: Vec::new(),
            node_sizes: Vec::new(),
            node_types: Vec::new(),
            node_in_degree: Vec::new(),
            node_out_degree: Vec::new(),
            next_node_index: 0,
            weight_matrices: Vec::new(),
            edges: Vec::new(),
            nodes_map: BTreeMap::new(),
        }
    }
}

impl<N: Sub<Output = N> + MulAssign + Mul<Output = N> + Default + AddAssign + Sum + Copy + Debug + SubAssign, A: ActivationFn<N>, NodeId: Eq + Ord + Clone> PCN<N, A, NodeId> {
    pub fn add_node(&mut self, id: &NodeId, activation_function: A, size: usize) {
        debug_assert!(!self.nodes_map.contains_key(id));

        let node_index = self.next_node_index;
        self.next_node_index += 1;

        self.activation_functions.push(activation_function);
        self.node_values.push(NodeValues::new(size));
        self.node_predictions.push(NodePredictions::new(size));
        self.node_gain_modulated_errors
            .push(GainModulatedErrors::new(size));
        self.node_errors.push(NodeErrors::new(size));
        self.node_sizes.push(size);
        self.node_types.push(Default::default());
        self.node_in_degree.push(0);
        self.node_out_degree.push(0);

        self.nodes_map.insert(id.clone(), node_index);
    }

    pub fn add_edge(&mut self, source_id: &NodeId, target_id: &NodeId) {
        debug_assert!(self.nodes_map.contains_key(source_id));
        debug_assert!(self.nodes_map.contains_key(target_id));
        debug_assert!(source_id != target_id);

        let source = self.nodes_map.get(source_id).unwrap();
        let target = self.nodes_map.get(target_id).unwrap();

        let source_size = self.node_sizes[*source];
        let target_size = self.node_sizes[*target];
        let weight_matrix = WeightMatrix::LayerWeights(Box::new(RowWise::new(target_size, source_size)));
        let weight_matrix_index = self.weight_matrices.len();

        self.node_in_degree[*target] += 1;
        self.node_out_degree[*source] += 1;

        self.weight_matrices.push(weight_matrix);
        self.edges
            .push(Edge::new(*source, *target, weight_matrix_index));
    }

    fn reverse_lookup_node(&self, node_index: usize) -> &NodeId {
        for (k, v) in self.nodes_map.iter() {
            if *v == node_index {
                return k;
            }
        }

        panic!("node index {} not in PCN", node_index);
    }

    pub fn compute_errors(&mut self) {
        // TODO return accumulated error

        let iter = self
            .node_errors
            .iter_mut()
            .zip(&self.node_values)
            .zip(&self.node_predictions)
            .zip(&self.node_types)
            .zip(&self.node_in_degree);

        for ((((error, value), prediction), node_type), node_in_degree) in iter {
            if *node_in_degree == 0 {
                error.0.as_mut().fill(Default::default());
            } else {
                error.compute(*node_type, value, prediction);
            }
        }
    }

    pub fn inference_steps(&mut self, gamma: N, n: usize) /*-> N*/ {
        // TODO return accumulated error

        for _i in 0..n {
            self.compute_predictions();
            self.compute_errors();
            self.compute_values(gamma);
        }
    }

    pub fn compute_predictions(&mut self) {
        for (prediction, node_type) in self.node_predictions.iter_mut().zip(&self.node_types) {
            if node_type.update_predictions() {
                prediction.set_to_default();
            }
        }

        for edge in self.edges.iter() {
            let weights = &self.weight_matrices[edge.weight_matrix_index];
            let source = &self.node_values[edge.source];
            let target = &mut self.node_predictions[edge.target];

            if self.node_types[edge.target].update_predictions() {
                target.add_weighted_input(weights, source);
            }
        }

        for (i, prediction) in self.node_predictions.iter_mut().enumerate() {
            if self.node_types[i].update_predictions() {
                prediction.apply_activation_function(&self.activation_functions[i]);
            }
        }
    }

    pub fn compute_gain_modulated_errors(&mut self) {
        for gain_modulated_errors in self.node_gain_modulated_errors.iter_mut() {
            gain_modulated_errors.set_to_default();
        }

        for edge in self.edges.iter() {
            let weights = &self.weight_matrices[edge.weight_matrix_index];
            let source = &self.node_values[edge.source];
            let target = &mut self.node_gain_modulated_errors[edge.target];

            target.add_weighted_input(weights, source);
        }

        for (i, gain_modulated_errors) in self.node_gain_modulated_errors.iter_mut().enumerate() {
            gain_modulated_errors.compute(&self.activation_functions[i], &self.node_errors[i]);
        }
    }

    pub fn compute_values(&mut self, gamma: N) {
        self.compute_gain_modulated_errors();

        for ((e, v), t) in self
            .node_errors
            .iter()
            .zip(self.node_values.iter_mut())
            .zip(self.node_types.iter())
        {
            if t.update_values() {
                v.update_with_local_errors(gamma, e);
            }
        }

        for edge in self.edges.iter() {
            let w = &self.weight_matrices[edge.weight_matrix_index];
            let gme = &self.node_gain_modulated_errors[edge.target];
            let v = &mut self.node_values[edge.source];
            let t = &self.node_types[edge.source];

            if t.update_values() {
                v.update_with_gain_modulated_errors(gamma, w, gme);
            }
        }
    }

    pub fn learn_hebb(&mut self, alpha: N) {
        self.compute_gain_modulated_errors();

        for edge in self.edges.iter() {
            let w = &mut self.weight_matrices[edge.weight_matrix_index];
            let h = &self.node_gain_modulated_errors[edge.target];
            let x = &self.node_values[edge.source];

            w.learn_hebb(alpha, x, h);
        }
    }

    pub fn learn_oja(&mut self, _alpha: N) {
        todo!()
    }

    pub fn set_values(&mut self, node_id: &NodeId, values: &[N]) {
        let node_index = self.nodes_map.get(node_id).unwrap();
        self.node_values[*node_index]
            .0
            .as_mut()
            .copy_from_slice(values);
    }

    pub fn set_values_from_bool(&mut self, node_id: &NodeId, true_val: N, false_val: N, values: &[bool]) {
        let node_index = self.nodes_map.get(node_id).unwrap();
        let iter = self.node_values[*node_index].0.as_mut().iter_mut();

        for (i, v) in values.iter().zip(iter) {
            *v = if *i { true_val } else { false_val };
        }
    }

    pub fn set_predictions(&mut self, node_id: &NodeId, values: &[N]) {
        let node_index = self.nodes_map.get(node_id).unwrap();
        self.node_predictions[*node_index]
            .0
            .as_mut()
            .copy_from_slice(values);
    }

    pub fn fix_node(&mut self, node_id: &NodeId, values: &[N]) {
        self.set_node_type(node_id, NodeType::Label);
        self.set_values(node_id, values);
        self.set_predictions(node_id, values);
    }

    pub fn fix_node_from_bool(&mut self, node_id: &NodeId, true_val: N, false_val: N, values: &[bool]) {
        self.set_node_type(node_id, NodeType::Label);
        self.set_values_from_bool(node_id, true_val, false_val, values);
        self.set_predictions_from_bool(node_id, true_val, false_val, values);
    }

    pub fn set_predictions_from_bool(&mut self, node_id: &NodeId, true_val: N, false_val: N, values: &[bool]) {
        let node_index = self.nodes_map.get(node_id).unwrap();
        let iter = self.node_predictions[*node_index].0.as_mut().iter_mut();

        for (i, v) in values.iter().zip(iter) {
            *v = if *i { true_val } else { false_val };
        }
    }

    pub fn set_node_type(&mut self, node_id: &NodeId, node_type: NodeType) {
        let node_index = self.nodes_map.get(node_id).unwrap();
        self.node_types[*node_index] = node_type;
    }

    pub fn node_values(&self, node_id: &NodeId) -> &[N] {
        let node_index = self.nodes_map.get(node_id).unwrap();
        self.node_values[*node_index].0.as_ref()
    }
}

type NodeIndex = usize;

struct NodeValues<N>(Box<[N]>);

impl<N: Debug> Debug for NodeValues<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_list().entries(self.0.as_ref()).finish()
    }
}

impl<N: MulAssign + Mul<Output = N> + AddAssign + Sum + Default + Clone + Copy + SubAssign> NodeValues<N> {
    fn new(size: usize) -> Self {
        Self(vec![Default::default(); size].into_boxed_slice())
    }

    fn update_with_local_errors(&mut self, gamma: N, local_errors: &NodeErrors<N>) {
        for (v, e) in self.0.iter_mut().zip(local_errors.0.as_ref()) {
            *v -= gamma * *e;
        }
    }

    fn update_with_gain_modulated_errors(&mut self, gamma: N, weights: &WeightMatrix<N>, gain_modulated_errors: &GainModulatedErrors<N>) {
        weights.matrix().vec_mul_add_scale(gamma, &gain_modulated_errors.0, self.0.as_mut());
    }

    fn as_column_vec(&self) -> ColumnVector<'_, N> {
        ColumnVector::new(&self.0)
    }
}

struct NodePredictions<N>(Box<[N]>);

impl<N: MulAssign + Mul<Output = N> + AddAssign + Sum + Default + Clone + Copy + SubAssign> NodePredictions<N> {
    fn new(size: usize) -> Self {
        Self(vec![Default::default(); size].into_boxed_slice())
    }

    fn set_to_default(&mut self) {
        self.0.fill(Default::default());
    }

    fn add_weighted_input(&mut self, weights: &WeightMatrix<N>, source: &NodeValues<N>) {
        weights.matrix().mul_vec_add(&source.0, self.0.as_mut());
    }

    fn apply_activation_function<A: ActivationFn<N>>(&mut self, activation_function: &A) {
        activation_function.eval_inplace(self.0.as_mut());
    }
}

impl<N: Debug> Debug for NodePredictions<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_list().entries(self.0.as_ref()).finish()
    }
}

struct GainModulatedErrors<N>(Box<[N]>);

impl<N: Default + Clone> GainModulatedErrors<N> {
    fn new(size: usize) -> Self {
        Self(vec![Default::default(); size].into_boxed_slice())
    }

    fn set_to_default(&mut self) {
        self.0.as_mut().fill(Default::default());
    }

    fn as_row_vec<'a>(&'a self) -> RowVector<'a, N> {
        RowVector::new(&self.0)
    }
}

impl<N: AddAssign + MulAssign + Mul<Output = N> + Sum + Copy + Default + SubAssign> GainModulatedErrors<N> {
    fn add_weighted_input(&mut self, weights: &WeightMatrix<N>, source: &NodeValues<N>) {
        weights.matrix().mul_vec_add(source.0.as_ref(), self.0.as_mut());
    }

    fn compute<A: ActivationFn<N>>(&mut self, activation_function: &A, errors: &NodeErrors<N>) {
        activation_function.diff_inplace_mul(errors.0.as_ref(), self.0.as_mut());
    }
} 

struct NodeErrors<N>(Box<[N]>);

impl<N: Debug> Debug for NodeErrors<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_list().entries(self.0.as_ref()).finish()
    }
}

impl<N: Default + Clone> NodeErrors<N> {
    fn new(size: usize) -> Self {
        Self(vec![Default::default(); size].into_boxed_slice())
    }
}

impl<N: Sub<Output = N> + Copy> NodeErrors<N> {
    fn compute(&mut self, node_type: NodeType, values: &NodeValues<N>, predictions: &NodePredictions<N>) {
        if node_type.is_label() {
            for ((v, p), e) in values.0.iter().zip(predictions.0.iter()).zip(self.0.iter_mut()) {
                *e = *p - *v;
            }
        } else {
            for ((v, p), e) in values.0.iter().zip(predictions.0.iter()).zip(self.0.iter_mut()) {
                *e = *v - *p;
            }
        }
    }
}

struct Edge {
    source: NodeIndex,
    target: NodeIndex,
    weight_matrix_index: usize,
}

impl Edge {
    fn new(source: NodeIndex, target: NodeIndex, weight_matrix_index: usize) -> Self {
        Self {
            source,
            target,
            weight_matrix_index,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Default, Debug)]
pub enum NodeType {
    #[default]
    Internal,
    Sensor,
    Label,
}

impl NodeType {
    pub fn update_predictions(&self) -> bool {
        *self != Self::Label
    }

    pub fn is_label(&self) -> bool {
        *self == Self::Label
    }

    pub fn update_values(&self) -> bool {
        *self == Self::Internal
    }
}

enum WeightMatrix<N: MulAssign + Mul<Output = N> + Default + AddAssign + Sum + Copy> {
    LayerWeights(Box<RowWise<N>>),
    #[allow(unused)] // TODO allow for recurrent edges (from node to same node)
    HopfieldWeights(Box<Symmetric<N>>),
}

impl<N: MulAssign + Mul<Output = N> + Default + AddAssign + Sum + Copy + SubAssign> WeightMatrix<N> {
    fn matrix(&self) -> &dyn Matrix<N> {
        match self {
            WeightMatrix::LayerWeights(matrix) => matrix.as_ref(),
            WeightMatrix::HopfieldWeights(matrix) => matrix.as_ref(),
        }
    }

    fn learn_hebb(&mut self, alpha: N, values: &NodeValues<N>, gain_modulated_errors: &GainModulatedErrors<N>) {
        let values = values.as_column_vec();
        let gme = gain_modulated_errors.as_row_vec();
        let delta = values * gme;

        match self {
            WeightMatrix::LayerWeights(matrix) => add_scaled(matrix.as_mut(), alpha, &delta),
            WeightMatrix::HopfieldWeights(matrix) => add_scaled(matrix.as_mut(), alpha, &delta),
        }
    }
}

impl<N: Debug + MulAssign + Mul<Output = N> + Default + AddAssign + Sum + Copy> Debug for WeightMatrix<N> {
    fn fmt(&self, _fmt: &mut Formatter) -> Result<(), FmtError> {
        todo!()
    }
}
