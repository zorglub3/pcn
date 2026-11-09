use pcn::algorithms::*;
use pcn::randomize::*;
use pcn::{Builder, FloatActivationFn, PCN};
use rand::rng;

type IdType = String;
type NType = f64;
type AType = FloatActivationFn<NType>;

const GATE_INPUT_NODE: &str = "Input";
const GATE_OUTPUT_NODE: &str = "Output";
const GATE_HIDDEN_1: &str = "Hidden1";
const GATE_HIDDEN_2: &str = "Hidden2";

const ALPHA: NType = 0.1;
const GAMMA: NType = 0.1;
const INFERENCE_STEPS: usize = 16;
const TRAINING_ROUNDS: usize = 1000;

fn build_network() -> PCN<NType, AType, IdType> {
    use FloatActivationFn::Tanh;

    let builder: Builder<IdType, NType, AType> = Default::default();

    builder
        .add_node(GATE_INPUT_NODE.to_string(), Tanh(Default::default()), 2)
        .add_node(GATE_HIDDEN_1.to_string(), Tanh(Default::default()), 2)
        .add_node(GATE_HIDDEN_2.to_string(), Tanh(Default::default()), 2)
        .add_node(GATE_OUTPUT_NODE.to_string(), Tanh(Default::default()), 1)
        .add_edge(GATE_OUTPUT_NODE.to_string(), GATE_HIDDEN_2.to_string())
        .add_edge(GATE_HIDDEN_2.to_string(), GATE_HIDDEN_1.to_string())
        .add_edge(GATE_HIDDEN_1.to_string(), GATE_INPUT_NODE.to_string())
        .build()
}

fn make_batch(data: &[(Vec<NType>, Vec<NType>)]) -> Vec<LabeledData<IdType, NType>> {
    let mut result = Vec::new();

    for datum in data {
        result.push(LabeledData {
            sensor_patterns: vec![(GATE_INPUT_NODE.to_string(), datum.0.clone())],
            label_patterns: vec![(GATE_OUTPUT_NODE.to_string(), datum.1.clone())],
        });
    }

    result
}

fn error_func(target: &[NType], actual: &[NType]) -> f64 {
    let mut acc = 0.;

    println!("computing error: {:?} vs {:?}", target, actual);

    for (a, b) in target.iter().zip(actual) {
        let e = a - b;
        acc += e * e;
    }

    acc
}

fn error_to_f64(e: &[f64]) -> f64 {
    let mut acc = 0.;

    for v in e.iter() {
        acc += v * v;
    }

    acc
}

#[test]
fn test_and_gate() {
    let mut rng = rng();
    let mut pcn = build_network();
    let training_data = vec![
        (vec![1., 1.], vec![1.]),
        (vec![1., -1.], vec![-1.]),
        (vec![-1., 1.], vec![-1.]),
        (vec![-1., -1.], vec![-1.]),
    ];
    let batch = make_batch(&training_data);

    pcn.randomize_weights(&mut rng);

    for _i in 0..TRAINING_ROUNDS {
        let err = pcn.train_supervised(&batch, ALPHA, GAMMA, INFERENCE_STEPS, &error_to_f64);
        println!("error at {} is {}", _i, err);
    }

    let error = pcn.evaluate(&batch, GAMMA, INFERENCE_STEPS, &error_func);

    println!("error: {}", error);

    assert!(error < 0.1);
}
