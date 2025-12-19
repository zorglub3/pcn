use tensorflow::ops;
use tensorflow::Scope;
use tensorflow::Variable;
use tensorflow::Operation;
use tensorflow::Status;
use tensorflow::DataType;
// use tensorflow::Shape;
use tensorflow::Output;
use crate::PCNode;
use crate::ActivationFn;

pub enum TFError {
    TFStatus(Status),
}

impl From<Status> for TFError {
    fn from(status: Status) -> Self {
        Self::TFStatus(status)
    }
}

#[derive(Clone)]
pub struct NodeVariables {
    pub predictions: Variable,
    pub errors: Variable,
    pub values: Variable,
}

fn make_node_variables(
    node_size: u32,
    scope: &mut Scope,
) -> Result<NodeVariables, TFError> {
    let predictions = Variable::builder()
        .data_type(DataType::Float)
        .shape([node_size])
        .build(scope)?;

    let errors = Variable::builder()
        .data_type(DataType::Float)
        .shape([node_size])
        .build(scope)?;

    let values = Variable::builder()
        .data_type(DataType::Float)
        .shape([node_size])
        .build(scope)?;

    Ok(NodeVariables {
        predictions,
        errors,
        values,
    })
}

pub struct EdgeVariables {
    pub weights: Variable,
}

fn make_edge_variables(
    source_size: u32,
    target_size: u32,
    scope: &mut Scope,
) -> Result<EdgeVariables, TFError> {
    let weights = Variable::builder()
        .data_type(DataType::Float)
        .shape([source_size, target_size])
        .build(scope)?;

    Ok(EdgeVariables { weights })
}

fn activation_op<O: Into<Output>>(
    activation_fn: &ActivationFn,
    input: O,
    scope: &mut Scope,
) -> Result<Operation, TFError> {
    todo!()
}

fn activation_diff_op<O: Into<Output>>(
    activation_fn: &ActivationFn,
    input: O,
    scope: &mut Scope,
) -> Result<Operation, TFError> {
    todo!()
}

fn node_compute_predictions(
    node: usize, 
    node_activation: &[ActivationFn],
    node_variables: &[NodeVariables],
    edge_variables: &[EdgeVariables],
    targets: &[(usize, usize)], // (edge_index, node_index)
    scope: &mut Scope,
) -> Result<Operation, TFError> {
    let mut target_ops = Vec::new();

    for t in targets {
        target_ops.push(
            ops::mat_mul(
                edge_variables[t.0].weights.output().clone(),
                activation_op(
                    &node_activation[t.1],
                    node_variables[t.1].values.output().clone(),
                    scope,
                )?,
                scope,
            )?.into()
        );
    }

    let sum = ops::AddN::new().build_instance(target_ops, scope)?.op;

    ops::assign(
        node_variables[node].predictions.output().clone(),
        sum,
        scope,
    ).map_err(TFError::from)
}

fn node_compute_errors(
    node_variables: &NodeVariables,
    scope: &mut Scope,
) -> Result<Operation, TFError> {
    ops::assign(
        node_variables.errors.output().clone(),
        ops::sub(
            node_variables.values.output().clone(),
            node_variables.predictions.output().clone(),
            scope
        )?,
        scope,
    ).map_err(TFError::from)
}

fn node_compute_values(
    gamma: f64,
    node: usize,
    node_activation: &[ActivationFn],
    node_variables: &[NodeVariables],
    edge_variables: &[EdgeVariables],
    sources: &[(usize, usize)], // (edge index, node index)
    scope: &mut Scope,
) -> Result<Operation, TFError> {
    let mut source_ops = Vec::new();

    for s in sources {
        source_ops.push(
            ops::mat_mul(
                node_variables[s.1].errors.output().clone(),
                edge_variables[s.0].weights.output().clone(),
                scope,
            )?.into()
        );
    }

    let sum = ops::AddN::new().build_instance(source_ops, scope)?.op;

    ops::assign(
        node_variables[node].values.output().clone(),
        ops::mul(
            ops::constant(gamma, scope)?,
            ops::sub(
                activation_diff_op(
                    &node_activation[node],
                    sum,
                    scope,
                )?,
                node_variables[node].errors.output().clone(),
                scope,
            )?,
            scope,
        )?,
        scope,
    ).map_err(TFError::from)
}
