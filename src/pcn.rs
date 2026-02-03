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

pub struct PCNode {
    activation_fn: ActivationFn,
    // TODO change to boxed slice
    values: Vec<f64>,
    predictions: Vec<f64>,
    errors: Vec<f64>,
    fix_values: bool,
    tags: Vec<String>,
}

impl PCNode {
    pub fn new(activation_fn: ActivationFn, size: usize, str_tags: &[&str]) -> Self {
        let values = vec![0.; size];
        let predictions = vec![0.; size];
        let errors = vec![0.; size];
        let mut tags = Vec::new();

        for str_tag in str_tags {
            tags.push(str_tag.to_string());
        }

        PCNode {
            activation_fn,
            values,
            predictions,
            errors,
            fix_values: false,
            tags,
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    #[inline]
    pub fn errors(&self) -> &[f64] {
        &self.errors
    }

    #[inline]
    pub fn predictions(&self) -> &[f64] {
        &self.predictions
    }

    #[inline]
    pub fn activation(&self) -> &ActivationFn {
        &self.activation_fn
    }

    #[inline]
    pub fn set_values(&mut self, new_values: &[f64]) {
        if !self.fix_values {
            self.values.copy_from_slice(new_values);
        }
    }

    #[inline]
    pub fn update_values(&mut self, delta: &[f64]) {
        if !self.fix_values {
            add_inplace(delta, &mut self.values);
        }
    }

    #[inline]
    pub fn set_predictions(&mut self, new_predictions: &[f64]) {
        self.predictions.copy_from_slice(new_predictions);
    }

    #[inline]
    pub fn compute_errors(&mut self) -> f64 {
        let mut error_square_sum = 0.;

        for i in 0..self.size() {
            let err = self.values[i] - self.predictions[i];
            self.errors[i] = err;
            error_square_sum += err * err;
        }

        error_square_sum
    }

    #[inline]
    #[allow(dead_code)]
    pub fn compute_errors_with_variance(&mut self, sigma: f64) -> f64 {
        let mut error_square_sum = 0.;

        for i in 0..self.size() {
            let err = (self.values[i] - self.predictions[i]) / sigma;
            self.errors[i] = err;
            error_square_sum += err * err;
        }

        error_square_sum
    }

    #[inline]
    pub fn reset(&mut self) {
        self.values.fill(0.);
        self.errors.fill(0.);
        self.predictions.fill(0.);
        self.fix_values = false;
    }

    #[inline]
    pub fn fix_values(&mut self, new_values: &[f64]) {
        self.values.copy_from_slice(new_values);
        self.fix_values = true;
    }

    #[inline]
    pub fn randomize_values(&mut self, amount: f64, rng: &mut impl Rng) {
        self.fix_values = false;
        self.values.fill_with(|| rng.random_range(-amount..amount));
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
                .activation()
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
            err_sum_sqr += node_weight.compute_errors();
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
                w2.activation().diff(&a, &mut b); // b = f'(W x_source)
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
        self.compute_errors()
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
            target_node.activation().diff(&a, &mut b); // b = f'(W x_source)
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

    pub fn node_values(&self, id: NodeId) -> &[f64] {
        self.graph
            .node_weight(*self.nodes_map.get(&id).unwrap())
            .unwrap()
            .values()
    }

    pub fn node_predictions(&self, id: NodeId) -> &[f64] {
        self.graph
            .node_weight(*self.nodes_map.get(&id).unwrap())
            .unwrap()
            .predictions()
    }

    pub fn set_node_values(&mut self, id: NodeId, new_values: &[f64]) {
        self.graph
            .node_weight_mut(*self.nodes_map.get(&id).unwrap())
            .unwrap()
            .set_values(new_values);
    }

    pub fn fix_node_values(&mut self, id: NodeId, new_values: &[f64]) {
        self.graph
            .node_weight_mut(*self.nodes_map.get(&id).unwrap())
            .unwrap()
            .fix_values(new_values);
    }

    fn update_node_values(&mut self, index: &NodeIdx, delta: &[f64]) {
        self.graph
            .node_weight_mut(*index)
            .unwrap()
            .update_values(delta);
    }

    fn node_errors(&self, index: &NodeIdx) -> &[f64] {
        self.graph.node_weight(*index).unwrap().errors()
    }

    pub fn randomize_all_nodes(&mut self, amount: f64, rng: &mut impl Rng) {
        debug_assert!(amount > std::f64::EPSILON);

        for node_weight in self.graph.node_weights_mut() {
            node_weight.randomize_values(amount, rng);
        }
    }

    pub fn reset_all_nodes(&mut self) {
        for node_weight in self.graph.node_weights_mut() {
            node_weight.reset();
        }
    }

    /*
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
    */
}
