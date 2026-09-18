use pcn::{PCN, Builder, FloatActivationFn};
use rand::rng;

type IdType = usize;
type NType = f64;
type AType = FloatActivationFn<NType>;

const GATE_INPUT_NODE: usize = 0;
const GATE_OUTPUT_NODE: usize = 1;
const GATE_HIDDEN_1: usize = 2;
const GATE_HIDDEN_2: usize = 3;

const ALPHA: NType = 0.2;
const GAMMA: NType = 0.2;
const INFERENCE_STEPS: usize = 8;
const TRAINING_ROUNDS: uszie = 100;

fn build_network() -> PCN<NType, AType, IdType> {
    let builder: Builder<IdType, NType, AType> = Default::default();

    todo!() 
}

fn make_batch(data: &[(Vec<NType>, Vec<NType>)]) -> Vec<LabeledData<IdType, NType>> {
    let mut result = Vec::new();

    for datum in data {
        result.push(
            LabeledData {
                sensor_patterns: vec![(GATE_INPUT_NODE, datum.0.clone())],
                label_patters: vec![(GATE_OUTPUT_NODE, daatum.1.clone())],
            }
        );
    }

    result
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
    let batch = make_match(&training_data);

    for _i in 0..TRAINING_ROUNDS { 
        pcn.train_supervised(rng, batch, ALPHA, GAMMA, INFERENCE_STEPS);
    }

    let error = pcn.evaluate(rng, batch, GAMMA, INFERENCE_STEPS);

    assert!(error < 0.1);
}
