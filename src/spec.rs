use crate::activation::ActivationFn;
use crate::dmatrix::DMatrix;
use crate::pcn::LearningRule;
use crate::pcn::PCEdge;
use crate::pcn::PCNode;
use crate::pcn::WeightMatrix;
use crate::pcn::PCN;
use petgraph::graph::Graph;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::prelude::*;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct NodeId(usize);

impl NodeId {
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct MatrixId(usize);

impl MatrixId {
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }
}

#[derive(Serialize, Deserialize)]
struct Node {
    function: ActivationFn,
    size: usize,
    tags: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct Edge {
    to: NodeId,
    from: NodeId,
    matrix_id: MatrixId,
}

#[derive(Serialize, Deserialize, Default)]
pub struct Spec {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    matrices: Vec<WeightMatrix>,
}

impl Spec {
    pub fn nodes_size(&self) -> usize {
        self.nodes.len()
    }

    pub fn edges_size(&self) -> usize {
        self.edges.len()
    }

    pub fn matrices_size(&self) -> usize {
        self.matrices.len()
    }

    pub fn add_node(&mut self, size: usize, function: ActivationFn) -> NodeId {
        let index = self.nodes.len();

        self.nodes.push(Node {
            function,
            size,
            tags: Vec::new(),
        });

        NodeId::new(index)
    }

    pub fn add_node_with_tags(
        &mut self,
        size: usize,
        function: ActivationFn,
        tags: &[&str],
    ) -> NodeId {
        let tags = tags.iter().map(|t| t.to_string()).collect();
        let index = self.nodes.len();

        self.nodes.push(Node {
            function,
            size,
            tags,
        });

        NodeId::new(index)
    }

    pub fn add_weight_matrix(&mut self, from_size: usize, to_size: usize) -> MatrixId {
        let index = self.matrices.len();

        let weight_matrix = WeightMatrix {
            matrix: DMatrix::new(to_size, from_size, 0.),
            learning_rule: LearningRule::Oja,
        };

        self.matrices.push(weight_matrix);

        MatrixId::new(index)
    }

    pub fn add_random_weight_matrix(
        &mut self,
        from_size: usize,
        to_size: usize,
        rng: &mut impl Rng,
    ) -> MatrixId {
        let index = self.matrices.len();

        let mut matrix = DMatrix::new(to_size, from_size, 0.);
        for row in 0..matrix.rows() {
            for col in 0..matrix.cols() {
                matrix[(row, col)] = rng.random_range(-1. ..1.);
            }
        }

        let weight_matrix = WeightMatrix {
            matrix,
            learning_rule: LearningRule::Oja,
        };

        self.matrices.push(weight_matrix);

        MatrixId::new(index)
    }

    pub fn add_edge_with_matrix(&mut self, from: NodeId, to: NodeId, matrix_id: MatrixId) {
        debug_assert!(from.0 < self.nodes.len());
        debug_assert!(to.0 < self.nodes.len());
        debug_assert!(matrix_id.0 < self.matrices.len());

        let from_node = &self.nodes[from.0];
        let to_node = &self.nodes[to.0];
        let matrix_node = &self.matrices[matrix_id.0];

        if from_node.size != matrix_node.matrix.cols() || to_node.size != matrix_node.matrix.rows()
        {
            panic!("add_ege_with_matrix: mimatched matrix size");
        }

        let edge_data = Edge {
            from,
            to,
            matrix_id,
        };

        self.edges.push(edge_data);
    }

    pub fn add_edge(&mut self, from: NodeId, to: NodeId) {
        debug_assert!(from.0 < self.nodes.len());
        debug_assert!(to.0 < self.nodes.len());

        let from_node = &self.nodes[from.0];
        let to_node = &self.nodes[to.0];

        let matrix_id = self.add_weight_matrix(from_node.size, to_node.size);

        let edge_data = Edge {
            from,
            to,
            matrix_id,
        };

        self.edges.push(edge_data);
    }

    pub fn randomize_matrix(&mut self, matrix_id: MatrixId, amount: f64, rng: &mut impl Rng) {
        debug_assert!(matrix_id.0 < self.matrices.len());

        self.matrices[matrix_id.0].matrix.randomize(amount, rng);
    }

    pub fn randomize_all_matrices(&mut self, amount: f64, rng: &mut impl Rng) {
        for weight_matrix in self.matrices.iter_mut() {
            weight_matrix.matrix.randomize(amount, rng);
        }
    }

    pub fn randomize_all_matrices_xavier(&mut self, rng: &mut impl Rng) {
        for weight_matrix in self.matrices.iter_mut() {
            let amount = (6.
                / (weight_matrix.matrix.rows() as f64 + weight_matrix.matrix.cols() as f64))
                .sqrt();
            weight_matrix.matrix.randomize(amount, rng);
        }
    }

    pub fn build_model(&self) -> PCN {
        let mut graph = Graph::<PCNode, PCEdge>::new();
        let mut nodes_map = HashMap::new();
        let matrices = self.matrices.clone();

        for index in 0..self.nodes.len() {
            let node_id = NodeId(index);
            let node = &self.nodes[index];
            let graph_node = PCNode::new(
                node.function,
                node.size,
                &node.tags.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            );

            let node_index = graph.add_node(graph_node);

            nodes_map.insert(node_id, node_index);
        }

        for edge in &self.edges {
            let node_source = nodes_map.get(&edge.from).unwrap();
            let node_target = nodes_map.get(&edge.to).unwrap();

            graph.add_edge(*node_source, *node_target, PCEdge::new(edge.matrix_id.0));
        }

        PCN::new(graph, nodes_map, matrices)
    }

    pub fn save_model(&self, filename: &str) -> std::io::Result<()> {
        let json_str = serde_json::to_string(self).unwrap();
        let mut output = std::fs::File::create(filename)?;
        write!(output, "{}", json_str)
    }

    pub fn load_model(filename: &str) -> std::io::Result<Self> {
        let json_str = std::fs::read_to_string(filename)?;

        let spec: Self = serde_json::from_str(&json_str).unwrap();

        Ok(spec)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::activation::ActivationFn;

    #[test]
    fn default_spec_is_empty() {
        let spec = Spec::default();

        assert_eq!(spec.nodes_size(), 0);
        assert_eq!(spec.edges_size(), 0);
        assert_eq!(spec.matrices_size(), 0);
    }

    #[test]
    fn adding_edge_creates_a_matrix() {
        let mut spec = Spec::default();

        let n1 = spec.add_internal_node(3, ActivationFn::Tanh);
        let n2 = spec.add_internal_node(4, ActivationFn::Tanh);

        spec.add_edge(n1, n2);

        assert_eq!(spec.nodes_size(), 2);
        assert_eq!(spec.edges_size(), 1);
        assert_eq!(spec.matrices_size(), 1);
    }
}
