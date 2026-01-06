use crate::activation::ActivationFn;
use crate::pcn::PCEdge;
use crate::pcn::PCNode;
use crate::pcn::PCN;
use petgraph::graph::Graph;
use std::collections::HashMap;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum NodeMode {
    Sensor,
    Output,
    Hidden,
}

struct Node {
    function: ActivationFn,
    mode: NodeMode,
    size: usize,
}

struct Edge {
    to: NodeId,
    from: NodeId,
}

pub struct Spec {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

impl Spec {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
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

    pub fn add_edge(&mut self, from: NodeId, to: NodeId) {
        assert!(from.0 < self.nodes.len());
        assert!(to.0 < self.nodes.len());

        let from_node = &self.nodes[from.0];

        if from_node.mode == NodeMode::Sensor {
            panic!("add_edge: source node cannot be a sensor");
        }

        let edge_data = Edge { from, to };

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
}
