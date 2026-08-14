use crate::activation::ActivationFn;
use crate::builder::Builder;

pub trait CentralNode<NodeId: Eq + Ord + Clone> {
    fn add_central_node(self, node_size: usize, sensors: &[NodeId], labels: &[NodeId]) -> Self;
}

impl<NodeId: Eq + Ord + Clone> CentralNode<NodeId> for Builder<NodeId> {
    fn add_central_node(self, node_size: usize, sensors: &[NodeId], labels: &[NodeId]) -> Self {
        let mut builder = self;
        let central_node = builder.new_node_id();

        builder = builder.add_node(central_node.clone(), ActivationFn::Tanh, node_size);

        for sensor in sensors {
            builder = builder.add_edge(sensor.clone(), central_node.clone());
        }

        for label in labels {
            builder = builder.add_edge(central_node.clone(), label.clone());
        }

        builder
    }
}
