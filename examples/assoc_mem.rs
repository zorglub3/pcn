use pcn::PCN;

const SENSOR_SIZE: usize = 128;
const INTERNAL_SIZE: usize = 64;
const MEMORY_SIZE: usize = 16;
const GAMMA: f64 = 0.01;
const ALPHA: f64 = 0.01;
const INFERENCE_STEPS: usize = 1000;

fn main() {
    println!("Creating associative memory");
    let mut assoc_mem = PCN::new();

    let sensor = assoc_mem.add_sensor_node(SENSOR_SIZE);
    let hidden = assoc_mem.add_internal_node(INTERNAL_SIZE);
    let memory = assoc_mem.add_memory_node(MEMORY_SIZE);

    assoc_mem
        .connect_nodes(&sensor, &hidden)
        .expect("Could not connect nodes");
    assoc_mem
        .connect_nodes(&hidden, &memory)
        .expect("Could not connect nodes");

    println!("Learning a pattern");
    let pattern: Vec<f64> = vec![0.; SENSOR_SIZE];
    let learning_mask: Vec<bool> = vec![true; SENSOR_SIZE];
    let mem_pattern: Vec<f64> = vec![0.; MEMORY_SIZE];

    assoc_mem
        .set_sensor_values(&sensor, &pattern, &learning_mask)
        .expect("Could not set sensor layer for learning");
    assoc_mem
        .set_memory_values(&memory, &mem_pattern)
        .expect("Could not set memory layer");
    assoc_mem
        .inference_steps(GAMMA, INFERENCE_STEPS)
        .expect("Could not perform inference step");
    assoc_mem
        .learning_step(ALPHA)
        .expect("Could not perform learning step");

    println!("Resetting network");

    println!("All done");
}
