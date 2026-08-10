use crate::activation::ActivationFn;
use crate::pcn::*;

struct NodeSpec<Id: Eq + Ord> {
    id: Id,
    size: usize,
    activation_function: ActivationFn,
}

#[derive(Eq, PartialEq)]
struct EdgeSpec<Id: Eq + Ord> {
    source: Id,
    target: Id,
}

pub struct Builder<Id: Eq + Ord + Clone> {
    nodes: Vec<NodeSpec<Id>>,
    edges: Vec<EdgeSpec<Id>>,
    node_id_source: Option<Box<dyn IdSource<Id>>>,
}

impl<Id: Eq + Ord + Clone> Default for Builder<Id> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_id_source: None,
        }
    }
}

#[allow(unused)]
impl<Id: Eq + Ord + Clone> Builder<Id> {
    pub fn with_generator(mut self, node_id_source: Box<dyn IdSource<Id>>) -> Self {
        self.node_id_source = Some(node_id_source);
        self
    }

    pub fn new_node_id(&mut self) -> Id {
        self.node_id_source.as_mut().unwrap().generate_node_id()
    }

    #[allow(unused)]
    fn has_node(&self, id: &Id) -> bool {
        self.nodes.iter().any(|node_spec| node_spec.id == *id)
    }

    pub fn add_node(mut self, id: Id, activation_function: ActivationFn, size: usize) -> Self {
        debug_assert!(!self.has_node(&id));

        self.nodes.push(NodeSpec {
            id,
            size,
            activation_function,
        });
        self
    }

    pub fn add_edge(mut self, source: Id, target: Id) -> Self {
        debug_assert!(self.has_node(&source));
        debug_assert!(self.has_node(&target));

        let edge_spec = EdgeSpec { source, target };
        debug_assert!(!self.edges.contains(&edge_spec));

        self.edges.push(edge_spec);
        self
    }

    pub fn add_edge_with_matrix(mut self, source: Id, target: Id) -> Self {
        todo!()
    }

    pub fn add_recurrent_edge(mut self, node: Id) -> Self {
        todo!()
    }

    pub fn add_recurrent_edge_with_matrix(mut self, node: Id, matrix: Id) -> Self {
        todo!()
    }

    pub fn add_matrix(mut self, id: Id, rows: usize, cols: usize) -> Self {
        todo!()
    }

    pub fn add_matrix_by_nodes(mut self, id: Id, source: Id, target: Id) -> Self {
        todo!()
    }

    pub fn build(self) -> PCN<Id> {
        let mut pcn: PCN<Id> = PCN::default();

        for node in &self.nodes {
            pcn.add_node(&node.id, node.activation_function, node.size);
        }

        for edge in &self.edges {
            pcn.add_edge(&edge.source, &edge.target);
        }

        pcn
    }
}

pub trait IdSource<Id: Eq + Ord + Clone> {
    fn generate_node_id(&mut self) -> Id;
}
