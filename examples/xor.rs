use pcn::ActivationFn;
use pcn::NodeId;
use pcn::Spec;
use pcn::PCN;
use rand::prelude::*;

const SENSOR_SIZE: usize = 1;
const INTERNAL_SIZE: usize = 8;
const MEMORY_SIZE: usize = 2;
const INTERNAL_LAYER_COUNT: usize = 4;

const GAMMA: f64 = 0.2;
const ALPHA: f64 = 0.2;
const INFERENCE_STEPS: usize = 16;
const LEARNING_STEPS: usize = 1000;
// (INTERNAL_SIZE * INTERNAL_LAYER_COUNT + SENSOR_SIZE + MEMORY_SIZE) * 200;
const TEST_TRIALS: usize = 20;

const F64_TRUE: f64 = 0.5;
const F64_FALSE: f64 = -0.5;

const TEST_PATTERNS: [([bool; 2], [bool; 1]); 4] = [
    ([true, true], [false]),
    ([true, false], [true]),
    ([false, true], [true]),
    ([false, false], [false]),
];

fn random_bool_pattern<const C: usize>(rng: &mut impl Rng) -> [bool; C] {
    let mut x = [true; C];
    for i in 0..C {
        x[i] = rng.random();
    }
    x
}

fn bool_to_f64<const C: usize>(pattern: &[bool; C]) -> [f64; C] {
    let mut x = [0.; C];
    for i in 0..C {
        x[i] = if pattern[i] { F64_TRUE } else { F64_FALSE };
    }
    x
}

fn xor_pattern(pattern: &[bool; 2]) -> [bool; 1] {
    [pattern[0] != pattern[1]]
}

fn test_it(memory: NodeId, sensor: NodeId, pcn: &mut PCN, rng: &mut impl Rng) -> f64 {
    let mut total_error = 0.;

    for _i in 0..TEST_TRIALS {
        for (bool_input_pattern, bool_output_pattern) in TEST_PATTERNS {
            let f64_input_pattern = bool_to_f64(&bool_input_pattern);
            let f64_output_pattern = bool_to_f64(&bool_output_pattern);

            pcn.reset_all_nodes();
            pcn.randomize_all_nodes(0.1, rng);
            pcn.set_memory_values(memory, &f64_input_pattern);
            pcn.set_sensor_values(sensor, &[0.], &[false]);

            pcn.inference_steps(GAMMA, INFERENCE_STEPS);

            let output = pcn.get_node_values(sensor)[0].tanh();
            let error = f64_output_pattern[0] - output;
            total_error += error * error;

            // println!(" - {:?} => {:?}", &f64_input_pattern, output);
            // println!(" - error: {}", error);
        }
    }

    // println!("Total error (sum of squares): {}", total_error);
    total_error / (TEST_TRIALS as f64)
}

fn main() {
    let mut rng = rand::rng();

    let mut spec = Spec::default();
    let sensor = spec.add_sensor_node(SENSOR_SIZE, ActivationFn::Tanh);
    let memory = spec.add_output_node(MEMORY_SIZE, ActivationFn::Tanh);

    let mut hidden_layers = Vec::new();
    for _i in 0..INTERNAL_LAYER_COUNT {
        hidden_layers.push(spec.add_internal_node(INTERNAL_SIZE, ActivationFn::Tanh));
    }

    for i in 0..INTERNAL_LAYER_COUNT - 1 {
        spec.add_edge(hidden_layers[i], hidden_layers[i + 1]);
    }
    spec.add_edge(hidden_layers[INTERNAL_LAYER_COUNT - 1], sensor);
    spec.add_edge(memory, hidden_layers[0]);

    spec.randomize_all_matrices_xavier(&mut rng);

    let mut xor = spec.build_model();

    let initial_error = test_it(memory, sensor, &mut xor, &mut rng);

    println!("# Pre learning test error: {}", initial_error);

    let mask = vec![true; SENSOR_SIZE];

    let mut count = 0;
    let mut error_acc = 0.;

    for i in 0..LEARNING_STEPS {
        let bool_input_pattern = random_bool_pattern::<2>(&mut rng);
        let f64_input_pattern = bool_to_f64(&bool_input_pattern);
        let bool_output_pattern = xor_pattern(&bool_input_pattern);
        let f64_output_pattern = bool_to_f64(&bool_output_pattern);

        // println!("pattern to learn {:?} => {:?}", f64_input_pattern, f64_output_pattern);
        xor.reset_all_nodes();
        xor.randomize_all_nodes(0.1, &mut rng);
        // xor.set_sensor_values(sensor, &f64_input_pattern, &[false, false]);
        // xor.set_memory_values(memory, &f64_output_pattern);
        // xor.inference_steps(GAMMA, INFERENCE_STEPS);
        // xor.pp();

        xor.set_memory_values(memory, &f64_input_pattern);
        xor.set_sensor_values(sensor, &f64_output_pattern, &mask);

        // xor.infer_and_learn(GAMMA, ALPHA, INFERENCE_STEPS);
        xor.inference_steps(GAMMA, INFERENCE_STEPS);
        xor.learning_step(ALPHA);
        // xor.pp();

        error_acc += xor.get_total_energy();
        count += 1;

        if i % 200 == 0 && count > 0 {
            println!(
                "learned {} samples, energy avg. is {}",
                i + 1,
                error_acc / (count as f64)
            );
            count = 0;
            error_acc = 0.;
            println!(" - test: {}", test_it(memory, sensor, &mut xor, &mut rng));
        }
    }
    println!("learned {} samples. done.", LEARNING_STEPS);

    let final_error = test_it(memory, sensor, &mut xor, &mut rng);
    println!("# Post learning testing error: {}", final_error);
    println!("# Error diff: {}", final_error - initial_error);
    println!(
        "# This is {}",
        if final_error - initial_error < 0. {
            "good"
        } else {
            "bad"
        }
    );

    /*
    let mut total_error = 0.;

    for (bool_input_pattern, bool_output_pattern) in TEST_PATTERNS {
        let f64_input_pattern = bool_to_f64(&bool_input_pattern);
        let f64_output_pattern = bool_to_f64(&bool_output_pattern);

        xor.reset_all_nodes();
        // xor.randomize_all_nodes(0.5, &mut rng);
        xor.set_memory_values(memory, &f64_input_pattern);
        xor.set_sensor_values(sensor, &[0.], &[false]);

        xor.inference_steps(GAMMA, INFERENCE_STEPS);

        let output = xor.get_node_values(sensor)[0].tanh();
        let error = f64_output_pattern[0] - output;
        total_error += error * error;

        println!(" - {:?} => {:?}", &f64_input_pattern, output);
        println!(" - error: {}", error);
    }

    println!("Total error (sum of squares): {}", total_error);
    */
}
