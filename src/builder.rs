use crate::activation::ActivationFn;
use crate::pcn::*;

struct NodeSpec<NodeId: Eq + Ord> {
    id: NodeId,
    size: usize,
    activation_function: ActivationFn,
}

#[derive(Eq, PartialEq)]
struct EdgeSpec<NodeId: Eq + Ord> {
    source: NodeId,
    target: NodeId,
}

pub struct Builder<NodeId: Eq + Ord + Clone> {
    nodes: Vec<NodeSpec<NodeId>>,
    edges: Vec<EdgeSpec<NodeId>>,
    node_id_source: Option<Box<dyn NodeIdSource<NodeId>>>,
}

impl<NodeId: Eq + Ord + Clone> Default for Builder<NodeId> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_id_source: None,
        }
    }
}

impl<NodeId: Eq + Ord + Clone> Builder<NodeId> {
    pub fn with_generator(mut self, node_id_source: Box<dyn NodeIdSource<NodeId>>) -> Self {
        self.node_id_source = Some(node_id_source);
        self
    }

    pub fn new_node_id(&mut self) -> NodeId {
        self.node_id_source.as_mut().unwrap().generate_node_id()
    }

    #[allow(unused)]
    fn has_node(&self, id: &NodeId) -> bool {
        self.nodes.iter().any(|node_spec| node_spec.id == *id)
    }

    pub fn add_node(mut self, id: NodeId, activation_function: ActivationFn, size: usize) -> Self {
        debug_assert!(!self.has_node(&id));

        self.nodes.push(NodeSpec {
            id,
            size,
            activation_function,
        });
        self
    }

    pub fn add_edge(mut self, source: NodeId, target: NodeId) -> Self {
        debug_assert!(self.has_node(&source));
        debug_assert!(self.has_node(&target));

        let edge_spec = EdgeSpec { source, target };
        debug_assert!(!self.edges.contains(&edge_spec));

        self.edges.push(edge_spec);
        self
    }

    pub fn build(self) -> PCN<NodeId> {
        let mut pcn: PCN<NodeId> = PCN::default();

        for node in &self.nodes {
            pcn.add_node(&node.id, node.activation_function, node.size);
        }

        for edge in &self.edges {
            pcn.add_edge(&edge.source, &edge.target);
        }

        pcn
    }
}

pub trait NodeIdSource<NodeId: Eq + Ord + Clone> {
    fn generate_node_id(&mut self) -> NodeId;
}
