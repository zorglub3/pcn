use crate::builder::Builder;
use crate::activation::ActivationFn;

pub trait FeedForward<NodeId: Eq + Ord + Clone> {
    fn add_feed_forward(self, sensor_node: NodeId, label_node: NodeId, hidden_layer_sizes: &[usize]) -> Self;
}

impl<NodeId: Eq + Ord + Clone> FeedForward<NodeId> for Builder<NodeId> {
    fn add_feed_forward(self, sensor_node: NodeId, label_node: NodeId, hidden_layer_sizes: &[usize]) -> Self {
        let mut hidden_nodes = Vec::new();
        let mut builder = self;

        for hidden_layer_size in hidden_layer_sizes {
            let node_id = builder.new_node_id();
            hidden_nodes.push(node_id.clone());
            builder = builder.add_node(node_id.clone(), ActivationFn::Tanh, *hidden_layer_size);
        }

        if !hidden_nodes.is_empty() {
            builder = builder.add_edge(sensor_node, hidden_nodes[0].clone());
            builder = builder.add_edge(hidden_nodes.last().unwrap().clone(), label_node);

            for i in 0..(hidden_nodes.len() - 1) {
                builder = builder.add_edge(hidden_nodes[i].clone(), hidden_nodes[i + 1].clone());
            }
        } else {
            builder = builder.add_edge(sensor_node, label_node);
        }

        builder
    }
}
