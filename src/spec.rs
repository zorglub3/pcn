use crate::dmatrix::DMatrix;
use crate::activation::ActivationFn;
use crate::pcn::PCEdge;
use crate::pcn::PCNode;
use crate::pcn::PCN;
use petgraph::graph::Graph;
use std::collections::HashMap;
use std::io::prelude::*;
use rand::Rng;
use serde::{Serialize, Deserialize};

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
#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub enum NodeMode {
    Sensor,
    Output,
    Hidden,
}

#[derive(Serialize, Deserialize)]
struct Node {
    function: ActivationFn,
    mode: NodeMode,
    size: usize,
}

#[derive(Serialize, Deserialize)]
struct Edge {
    to: NodeId,
    from: NodeId,
    matrix_id: MatrixId,
}

#[derive(Serialize, Deserialize)]
pub struct Spec {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    matrices: Vec<DMatrix<f64>>,
}

impl Spec {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            matrices: Vec::new(),
        }
    }

    fn add_node(&mut self, node: Node) -> NodeId {
        let index = self.nodes.len();

        self.nodes.push(node);

        NodeId::new(index)
    }

    pub fn add_sensor_node(&mut self, size: usize, function: ActivationFn) -> NodeId {
        self.add_node(Node {
            function,
            mode: NodeMode::Sensor,
            size,
        })
    }

    pub fn add_output_node(&mut self, size: usize, function: ActivationFn) -> NodeId {
        self.add_node(Node {
            function,
            mode: NodeMode::Output,
            size,
        })
    }

    pub fn add_internal_node(&mut self, size: usize, function: ActivationFn) -> NodeId {
        self.add_node(Node {
            function,
            mode: NodeMode::Hidden,
            size,
        })
    }

    pub fn add_weight_matrix(&mut self, from_size: usize, to_size: usize) -> MatrixId {
        let index = self.matrices.len();

        self.matrices.push(DMatrix::new(to_size, from_size, 0.));

        MatrixId::new(index)
    }

    pub fn add_random_weight_matrix(&mut self, from_size: usize, to_size: usize, rng: &mut impl Rng) -> MatrixId {
        let index = self.matrices.len();

        let mut matrix = DMatrix::new(to_size, from_size, 0.);
        for row in 0..matrix.rows() {
            for col in 0..matrix.cols() {
                matrix[(row, col)] = rng.random_range(-1. .. 1.);
            }
        }

        self.matrices.push(matrix);

        MatrixId::new(index)
    }

    pub fn add_edge_with_matrix(&mut self, from: NodeId, to: NodeId, matrix_id: MatrixId) {
        assert!(from.0 < self.nodes.len());
        assert!(to.0 < self.nodes.len());
        assert!(matrix_id.0 < self.matrices.len());

        let from_node = &self.nodes[from.0];
        let to_node = &self.nodes[to.0];
        let matrix_node = &self.matrices[matrix_id.0];

        if from_node.mode == NodeMode::Sensor {
            panic!("add_edge: source node cannot be a sensor");
        }

        if from_node.size != matrix_node.cols() || to_node.size != matrix_node.rows() {
            panic!("add_ege_with_matrix: mimatched matrix size");
        }

        let edge_data = Edge { from, to, matrix_id };

        self.edges.push(edge_data);
    }

    pub fn add_edge(&mut self, from: NodeId, to: NodeId) {
        assert!(from.0 < self.nodes.len());
        assert!(to.0 < self.nodes.len());

        let from_node = &self.nodes[from.0];
        let to_node = &self.nodes[to.0];

        if from_node.mode == NodeMode::Sensor {
            panic!("add_edge: source node cannot be a sensor");
        }

        let matrix_id = self.add_weight_matrix(from_node.size, to_node.size);

        let edge_data = Edge { from, to, matrix_id };

        self.edges.push(edge_data);
    }

    pub fn build_model(&self) -> PCN {
        let mut graph = Graph::<PCNode, PCEdge>::new();
        let mut nodes_map = HashMap::new();

        for index in 0..self.nodes.len() {
            let node_id = NodeId(index);
            let node = &self.nodes[index];
            let graph_node = match node.mode {
                NodeMode::Sensor => PCNode::new_sensor(node.size, node.function),
                NodeMode::Hidden => PCNode::new_internal(node.size, node.function),
                NodeMode::Output => PCNode::new_memory(node.size, node.function),
            };

            let node_index = graph.add_node(graph_node);

            nodes_map.insert(node_id, node_index);
        }

        for edge in &self.edges {
            let node_source = nodes_map.get(&edge.from).unwrap();
            let node_target = nodes_map.get(&edge.to).unwrap();

            let source_size = self.nodes[edge.from.0].size;
            let target_size = self.nodes[edge.to.0].size;

            graph.add_edge(
                *node_source,
                *node_target,
                PCEdge::new(source_size, target_size),
            );
        }

        PCN::new(graph, nodes_map)
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
