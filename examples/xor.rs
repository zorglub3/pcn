use pcn::ActivationFn::Tanh;
use pcn::Spec;
use rand::prelude::*;

const SENSOR_SIZE: usize = 3;
const INTERNAL_SIZE: usize = 3;
const MEMORY_SIZE: usize = 2;

const GAMMA: f64 = 0.1;
const ALPHA: f64 = 0.1;
const INFERENCE_STEPS: usize = 200;
const LEARNING_STEPS: usize = 1000;

fn rand_f64(rng: &mut impl Rng) -> f64 {
    let v = rng.random::<u32>() & 1;

    if v == 0 {
        -1.
    } else {
        1.
    }
}

fn main() {
    let mut rng = rand::rng();

    let mut spec = Spec::new();
    let sensor = spec.add_sensor_node(SENSOR_SIZE, Tanh);
    let hidden = spec.add_internal_node(INTERNAL_SIZE, Tanh);
    let hidden2 = spec.add_internal_node(INTERNAL_SIZE, Tanh);
    let memory = spec.add_output_node(MEMORY_SIZE, Tanh);

    spec.add_edge(hidden, sensor);
    spec.add_edge(hidden2, hidden);
    spec.add_edge(memory, hidden2);

    let mut xor = spec.build_model();

    let mask = vec![true; SENSOR_SIZE];

    for _i in 0..LEARNING_STEPS {
        xor.randomize_node(sensor, 1., &mut rng);
        xor.randomize_node(hidden, 1., &mut rng);
        xor.randomize_node(hidden2, 1., &mut rng);
        xor.randomize_node(memory, 1., &mut rng);

        let a = rand_f64(&mut rng);
        let b = rand_f64(&mut rng);
        let c = if a == b { -1. } else { 1. };
        let sensor_data = [a, b, c];
        let memory_data = [1., 1.];

        println!("learning: {:?}", &sensor_data);

        xor.set_sensor_values(sensor, &sensor_data, &mask);
        xor.set_layer_values(memory, &memory_data);

        xor.inference_steps(GAMMA, INFERENCE_STEPS);
        xor.learning_step(ALPHA);
    }

    xor.set_layer_values(hidden, &vec![0.; INTERNAL_SIZE]);
    xor.set_layer_values(memory, &vec![0.; MEMORY_SIZE]);
    xor.set_layer_values(sensor, &vec![0.; SENSOR_SIZE]);

    let test_mask = [true, true, false];

    xor.set_layer_values(memory, &[1., 1.]);
    xor.set_sensor_values(sensor, &vec![-1.,  1., 0.], &test_mask);
    xor.inference_steps(GAMMA, INFERENCE_STEPS);

    let node_values = xor.get_node_values(sensor).unwrap();

    println!("Got {:?}", node_values);
}

