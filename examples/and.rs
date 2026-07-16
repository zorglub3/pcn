use pcn::*;
use rand::prelude::*;

const GATE_INPUT_SIZE: usize = 2;
const GATE_OUTPUT_SIZE: usize = 1;

const GAMMA: f64 = 0.3;
const INFERENCE_STEPS: usize = 8;
const ALPHA: f64 = 0.3;
const LEARNING_STEPS: usize = 1000;

const TEST_PATTERNS: [([bool; 2], [bool; 1]); 4] = [
    ([true, true], [true]),
    ([false, true], [false]),
    ([true, false], [false]),
    ([false, false], [false]),
];

fn test_it(gate_input: NodeId, gate_output: NodeId, pcn: &mut PCN, rng: &mut impl Rng) -> f64 {
    let mut total_error = 0.;

    pcn.set_node_role(gate_output, NodeRole::Hidden);

    for (input_pattern, output_pattern) in TEST_PATTERNS {
        let f64_input_pattern = bool_to_f64(&input_pattern);
        let f64_output_pattern = bool_to_f64(&output_pattern);

        pcn.reset_all_nodes();
        pcn.randomize_all_nodes(0.1, rng);
        pcn.fix_node_values(gate_input, &f64_input_pattern, NodeRole::Memory);

        pcn.inference_steps(GAMMA, INFERENCE_STEPS);

        let error = square_error(&f64_output_pattern, pcn.node_values(gate_output));

        println!(
            "{:?} => {:?} gives error {} (values: {:?})",
            input_pattern,
            output_pattern,
            error,
            pcn.node_values(gate_output)
        );
        total_error += error;
    }

    total_error
}

fn learn_it(input_node: NodeId, output_node: NodeId, pcn: &mut PCN, rng: &mut impl Rng) {
    for _i in 0..LEARNING_STEPS {
        for (input_pattern, output_pattern) in TEST_PATTERNS {
            let f64_input_pattern = bool_to_f64(&input_pattern);
            let f64_output_pattern = bool_to_f64(&output_pattern);

            pcn.reset_all_nodes();
            pcn.randomize_all_nodes(1., rng);

            pcn.fix_node_values(input_node, &f64_input_pattern, NodeRole::Memory);
            pcn.fix_node_values(output_node, &f64_output_pattern, NodeRole::Sensor);

            pcn.inference_steps(GAMMA, INFERENCE_STEPS);
            pcn.learning_step(ALPHA);
        }
    }

    println!("After learning");
    pcn.pp();
}

fn main() {
    let mut rng = rand::rng();

    let mut spec = Spec::default();
    let gate_input = spec.add_node_with_tags(GATE_INPUT_SIZE, ActivationFn::Tanh, &["input"]);
    let gate_output = spec.add_node_with_tags(GATE_OUTPUT_SIZE, ActivationFn::Tanh, &["output"]);
    let bias = spec.add_node_with_tags(1, ActivationFn::Tanh, &["bias"]);

    spec.add_edge(bias, gate_output);
    spec.add_edge(gate_input, gate_output);

    spec.randomize_all_matrices_xavier(&mut rng);

    let mut and = spec.build_model();

    and.fix_node_values(bias, &[1.], NodeRole::Memory);

    println!("PRE LEARNING");
    let initial_error = test_it(gate_input, gate_output, &mut and, &mut rng);
    println!("Pre learning error: {}", initial_error);

    println!("LEARNING");
    learn_it(gate_input, gate_output, &mut and, &mut rng);

    println!("POST LEARNING");
    let final_error = test_it(gate_input, gate_output, &mut and, &mut rng);
    println!("Error after learning: {}", final_error);
}
