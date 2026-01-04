use crate::activation::ActivationFn;
use crate::pcn::PCN;
use petgraph::graph::Graph;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
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

    pub fn add_sensor_node(&mut self, size: usize, function: ActivationFn) -> NodeId {
        let index = self.nodes.len();

        let node_data = Node {
            function,
            mode: NodeMode::Sensor,
            size,
        };

        self.nodes.push(node_data);

        NodeId::new(index)
    }

    pub fn add_edge(&mut self, from: NodeId, to: NodeId) {
        if self.nodes.len() <= from.0 {
            panic!("add_edge: invalid from node id")
        };

        let to_node = if self.nodes.len() > to.0 {
            &self.nodes[to.0]
        } else {
            panic!("add_edge: invalid to node id")
        };

        if to_node.mode == NodeMode::Sensor {
            panic!("add_edge: target node cannot be a sensor");
        }

        let edge_data = Edge { from, to };

        self.edges.push(edge_data);
    }

    pub fn build_model(&self) -> PCN {
        todo!()
    }
}
