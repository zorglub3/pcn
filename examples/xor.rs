use pcn::ActivationFn::Tanh;
use pcn::Spec;
use rand::prelude::*;

const SENSOR_SIZE: usize = 1;
const INTERNAL_SIZE: usize = 90;
const MEMORY_SIZE: usize = 2;
const INTERNAL_LAYER_COUNT: usize = 4;

const GAMMA: f64 = 0.02;
const ALPHA: f64 = 0.1;
const INFERENCE_STEPS: usize = 32;
const LEARNING_STEPS: usize = 8000;
    // (INTERNAL_SIZE * INTERNAL_LAYER_COUNT + SENSOR_SIZE + MEMORY_SIZE) * 200;

const F64_TRUE: f64 = 0.75;
const F64_FALSE: f64 = -0.75;

const TEST_PATTERNS: [([bool; 2], [bool; 1]); 4] =
    [
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

fn main() {
    let mut rng = rand::rng();

    let mut spec = Spec::default();
    let sensor = spec.add_sensor_node(SENSOR_SIZE, Tanh);
    let memory = spec.add_output_node(MEMORY_SIZE, Tanh);

    let mut hidden_layers = Vec::new();
    for _i in 0..INTERNAL_LAYER_COUNT {
        hidden_layers.push(spec.add_internal_node(INTERNAL_SIZE, Tanh));
    }

    for i in 0..INTERNAL_LAYER_COUNT - 1 {
        spec.add_edge(hidden_layers[i], hidden_layers[i + 1]);
    }
    spec.add_edge(hidden_layers[INTERNAL_LAYER_COUNT - 1], sensor);
    spec.add_edge(memory, hidden_layers[0]);

    spec.randomize_all_matrices(0.5, &mut rng);

    let mut xor = spec.build_model();

    let mask = vec![true; SENSOR_SIZE];

    for i in 0..LEARNING_STEPS {
        let bool_input_pattern = random_bool_pattern(&mut rng);
        let f64_input_pattern = bool_to_f64(&bool_input_pattern);
        let bool_output_pattern = xor_pattern(&bool_input_pattern); 
        let f64_output_pattern = bool_to_f64(&bool_output_pattern);

        xor.reset_all_nodes();
        xor.set_memory_values(memory, &f64_input_pattern);
        xor.inference_steps(GAMMA, INFERENCE_STEPS);
        // xor.pp();

        xor.set_memory_values(memory, &f64_input_pattern);
        xor.set_sensor_values(sensor, &f64_output_pattern, &mask);

        xor.inference_steps(GAMMA, INFERENCE_STEPS);
        xor.learning_step(ALPHA);
        // xor.pp();

        if i % 100 == 0 {
            println!("learned {} samples", i);
            println!("- total energy {}", xor.get_total_energy());
        }
    }
    println!("learned {} samples. done.", LEARNING_STEPS);

    println!("testing...");

    let mut total_error = 0.;

    for (bool_input_pattern, bool_output_pattern) in TEST_PATTERNS {
        let f64_input_pattern = bool_to_f64(&bool_input_pattern);
        let f64_output_pattern = bool_to_f64(&bool_output_pattern);

        xor.reset_all_nodes();
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
}
