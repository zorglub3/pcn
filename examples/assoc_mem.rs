use pcn::PCN;
use pcn::ActivationFn;

const SENSOR_SIZE: usize = 8;
const INTERNAL_SIZE: usize = 4;
const MEMORY_SIZE: usize = 4;
const GAMMA: f64 = 0.1;
const ALPHA: f64 = 0.1;
const INFERENCE_STEPS: usize = 100;
const IL_STEPS: usize = 200;

fn generate_sensor_pattern(pattern_number: usize, pattern_values: &mut [f64]) {
    match pattern_number {
        0 => {
            for i in 0..pattern_values.len() {
                pattern_values[i] = if i % 2 == 0 || i % 5 == 0 {
                    1.
                } else {
                    -1.
                }
            }
        }
        _ => panic!("No pattern for {}", pattern_number),
    }
}

fn main() {
    println!("Creating associative memory");
    let mut assoc_mem = PCN::new();

    let sensor = assoc_mem.add_sensor_node(SENSOR_SIZE, ActivationFn::Tanh);
    let hidden = assoc_mem.add_internal_node(INTERNAL_SIZE, ActivationFn::Tanh);
    let memory = assoc_mem.add_memory_node(MEMORY_SIZE, ActivationFn::Tanh);

    assoc_mem
        .connect_nodes(&hidden, &sensor)
        .expect("Could not connect nodes");
    assoc_mem
        .connect_nodes(&memory, &hidden)
        .expect("Could not connect nodes");

    println!("Learning a pattern");
    let mut pattern: Vec<f64> = vec![0.; SENSOR_SIZE];
    let learning_mask: Vec<bool> = vec![true; SENSOR_SIZE];
    let mem_pattern: Vec<f64> = vec![0.; MEMORY_SIZE];

    generate_sensor_pattern(0, &mut pattern);

    assoc_mem
        .set_sensor_values(&sensor, &pattern, &learning_mask)
        .expect("Could not set sensor layer for learning");
    assoc_mem
        .set_layer_values(&memory, &mem_pattern)
        .expect("Could not set memory layer");

    for _i in 0..IL_STEPS {
        assoc_mem
            .inference_steps(GAMMA, INFERENCE_STEPS)
            .expect("Could not perform inference step");
        assoc_mem
            .learning_step(ALPHA)
            .expect("Could not perform learning step");
    }

    println!("Resetting network");
    assoc_mem
        .set_layer_values(&hidden, &vec![0.; INTERNAL_SIZE])
        .expect("Could not zero the internal node");
    assoc_mem
        .set_layer_values(&memory, &mem_pattern)
        .expect("Could not set memory node");
    assoc_mem
        .set_layer_values(&sensor, &vec![0.; SENSOR_SIZE])
        .expect("Could not zero the sensor layer");

    println!("Use network for recall");
    let mut partial_pattern = vec![0.; SENSOR_SIZE];
    let mut partial_mask = vec![true; SENSOR_SIZE];

    generate_sensor_pattern(0, &mut partial_pattern);

    for i in (SENSOR_SIZE/2)..SENSOR_SIZE {
        partial_pattern[i] = 0.;
        partial_mask[i] = false;
    }

    assoc_mem
        .set_sensor_values(&sensor, &partial_pattern, &partial_mask)
        .expect("Could not set partial sensor pattern");
    assoc_mem
        .inference_steps(GAMMA, INFERENCE_STEPS)
        .expect("Could not recall pattern");

    let node_values = assoc_mem
        .get_node_values(&sensor)
        .expect("Could not get sensor node values");

    println!("Original pattern: {:?}", &pattern);
    println!("Recalled pattern: {:?}", &node_values);

    println!("All done");
}
