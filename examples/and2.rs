use pcn::bool_to_f64;
use pcn::builder::*;
use pcn::pcn2::PCN;
use pcn::square_error;
use pcn::ActivationFn;
use rand::Rng;

const GATE_INPUT_SIZE: usize = 2;
const GATE_OUTPUT_SIZE: usize = 1;

const GAMMA: f64 = 0.3;
const INFERENCE_STEPS: usize = 8;
const ALPHA: f64 = 0.3;
const LEARNING_STEPS: usize = 1000;

const INPUT_NODE: usize = 0;
const OUTPUT_NODE: usize = 1;

const TEST_PATTERNS: [([bool; 2], [bool; 1]); 4] = [
    ([true, true], [true]),
    ([false, true], [false]),
    ([true, false], [false]),
    ([false, false], [false]),
];

type NodeId = usize;
type MyPCN = PCN<NodeId>;

fn test_it(gate_input: &NodeId, gate_output: &NodeId, pcn: &mut MyPCN) -> f64 {
    let mut total_error = 0.;

    for (input, output) in TEST_PATTERNS {
        pcn.set_values_from_bool(gate_input, &input);

        pcn.inference_steps(GAMMA, INFERENCE_STEPS);

        let output_pattern = bool_to_f64(&output);
        let err = square_error(&output_pattern, pcn.node_values(gate_output));
        total_error += err;
    }

    total_error
}

fn learn_it(
    gate_input: &NodeId,
    gate_output: &NodeId,
    pcn: &mut MyPCN,
    rng: &mut impl Rng,
    n: usize,
) {
    for _i in 0..n {
        for (input, output) in TEST_PATTERNS {
            pcn.randomize_values(rng);
            pcn.set_values_from_bool(gate_input, &input);
            pcn.set_values_from_bool(gate_output, &output);
            pcn.set_predictions_from_bool(gate_input, &input);

            pcn.inference_steps(GAMMA, INFERENCE_STEPS);
            pcn.learn_hebb(ALPHA);
        }
    }
}

fn main() {
    let mut rng = rand::rng();

    let builder = Builder::default();

    let mut pcn: MyPCN = builder
        .add_node(INPUT_NODE, ActivationFn::Tanh, GATE_INPUT_SIZE)
        .add_node(OUTPUT_NODE, ActivationFn::Tanh, GATE_OUTPUT_SIZE)
        .add_edge(INPUT_NODE, OUTPUT_NODE)
        .build();

    pcn.randomize_weights(&mut rng);

    let initial_error = test_it(&INPUT_NODE, &OUTPUT_NODE, &mut pcn);

    learn_it(
        &INPUT_NODE,
        &OUTPUT_NODE,
        &mut pcn,
        &mut rng,
        LEARNING_STEPS,
    );

    let final_error = test_it(&INPUT_NODE, &OUTPUT_NODE, &mut pcn);

    println!(
        "initial error: {} => final error: {}",
        initial_error, final_error
    );
}
