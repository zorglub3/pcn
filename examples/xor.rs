use pcn::ActivationFn::Tanh;
use pcn::Spec;
use rand::prelude::*;

const SENSOR_SIZE: usize = 3;
const INTERNAL_SIZE: usize = 10;
const MEMORY_SIZE: usize = 2;
const INTERNAL_LAYER_COUNT: usize = 4;

const GAMMA: f64 = 0.1;
const ALPHA: f64 = 0.1;
const INFERENCE_STEPS: usize = 10;
const LEARNING_STEPS: usize = 20000;
const LEARN_PER_SAMPLE: usize = 1;

#[allow(dead_code)]
fn random_xor_pattern(rng: &mut impl Rng) -> [f64; 3] {
    let a = rng.random::<bool>();
    let b = rng.random::<bool>();
    let c = if a == b { false } else { true };

    [
        if a { 1. } else { -1. },
        if b { 1. } else { -1. },
        if c { 1. } else { -1. },
    ]
}

fn random_pattern(rng: &mut impl Rng) -> [f64; 3] {
    let bool_pattern = random_bool_pattern(rng);
    bool_to_f64(&bool_pattern)
}

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
        x[i] = if pattern[i] { 1. } else { -1. };
    }
    x
}

fn is_xor_pattern(p: &[bool; 3]) -> bool {
    p[2] == !(p[0] == p[1])
}

fn main() {
    let mut rng = rand::rng();

    let mut spec = Spec::new();
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

    spec.randomize_all_matrices(1., &mut rng);

    let mut xor = spec.build_model();

    let mask = vec![true; SENSOR_SIZE];

    for i in 0..LEARNING_STEPS {
        let bp = random_bool_pattern::<3>(&mut rng);
        let is_xor = is_xor_pattern(&bp);
        let sensor_data = bool_to_f64(&bp);
        let mp = [is_xor, is_xor];
        let memory_data = bool_to_f64(&mp);
        // let sensor_data = random_pattern(&mut rng);
        // let memory_data = [1., 1.];

        // println!("learning: {:?} => {:?}", &memory_data, &sensor_data);

        for _k in 0..LEARN_PER_SAMPLE {
            for hidden in &hidden_layers {
                xor.randomize_node(*hidden, 0.1, &mut rng);
            }

            xor.set_sensor_values(sensor, &sensor_data, &mask);
            xor.set_layer_values(memory, &memory_data);

            // xor.inference_converge(GAMMA, 1.0);
            xor.inference_steps(GAMMA, INFERENCE_STEPS);
            // xor.infer_and_learn(GAMMA, ALPHA, INFERENCE_STEPS);

            // println!("final energy {}", xor.get_total_energy());

            xor.learning_step(ALPHA);
        }

        if i % 100 == 0 {
            println!("learned {} samples, energy {}", i, xor.get_total_energy());
        }

        // println!("=======================================");
    }
    println!("learned {} samples. done.", LEARNING_STEPS);

    let test_mask = [true, true, false];


    println!("testing...");
    for _x in 0..10 {
        for hidden in &hidden_layers {
            xor.randomize_node(*hidden, 0.01, &mut rng);
        }

        let pattern = random_pattern(&mut rng);

        println!("Before inference {:?}", &pattern);

        xor.set_layer_values(memory, &[1., 1.]);
        xor.set_sensor_values(sensor, &pattern, &test_mask);
        xor.inference_steps(0., 1);
        println!(" - energy before {}", xor.get_total_energy());
        xor.inference_steps(GAMMA, 320);

        let node_values = xor.get_node_values(sensor).unwrap();

        println!(" - got {:?}", &node_values);
        println!(" - energy after {}", xor.get_total_energy());
        // println!(" - Got {:?}", node_values.iter().map(|x| x.tanh()).collect::<Vec<f64>>());
    }
}
