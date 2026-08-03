use ::pcn::*;
use rand::Rng;

const SENSOR_SIZE: usize = 2;
const LABEL_SIZE: usize = 1;

const GAMMA: f64 = 0.5;
const INFERENCE_STEPS: usize = 4;
const ALPHA: f64 = 1.0;
const LEARNING_STEPS: usize = 1000;

const SENSOR_NODE: &str = "SENSOR";
const LABEL_NODE: &str = "LABEL";

const TEST_PATTERNS: [([bool; 2], [bool; 1]); 4] = [
    ([true, true], [true]),
    ([false, true], [false]),
    ([true, false], [false]),
    ([false, false], [false]),
];

type NodeId = String;
type MyPCN = PCN<NodeId>;

fn test_it(sensor_node: &NodeId, label_node: &NodeId, pcn: &mut MyPCN) -> f64 {
    let mut total_error = 0.;

    println!("test it!");
    for (input, output) in TEST_PATTERNS {
        println!(":: patterns {:?} => {:?}", input, output);
        pcn.set_node_type(sensor_node, NodeType::Sensor);
        pcn.set_node_type(label_node, NodeType::Internal);
        pcn.set_values_from_bool(sensor_node, &input);

        pcn.inference_steps(GAMMA, INFERENCE_STEPS);

        let output_pattern = bool_to_f64(&output);
        let err = square_error(&output_pattern, pcn.node_values(label_node));
        total_error += err;
        println!(" {:?} => {:?}", &input, pcn.node_values(label_node));
    }

    println!("testing done with error={}", total_error);

    total_error
}

fn learn_it(
    sensor_node: &NodeId,
    label_node: &NodeId,
    pcn: &mut MyPCN,
    rng: &mut impl Rng,
    n: usize,
) {
    let mut err = 0.;

    println!("Learning, {} steps", n);
    for _i in 0..n {
        for (input, output) in TEST_PATTERNS {
            //println!(":: patterns {:?} => {:?}", input, output);
            pcn.randomize_values(rng);
            pcn.set_node_type(sensor_node, NodeType::Sensor);
            pcn.set_values_from_bool(sensor_node, &input);
            pcn.fix_node_from_bool(label_node, &output);

            err = pcn.inference_steps(GAMMA, INFERENCE_STEPS);
            pcn.learn_hebb(ALPHA);
        }
    }

    println!("learning done. Final error: {}", err);
}

fn main() {
    let mut rng = rand::rng();

    let builder = Builder::default();

    let mut pcn: MyPCN = builder
        .add_node(SENSOR_NODE.to_string(), ActivationFn::Tanh, SENSOR_SIZE)
        .add_node(LABEL_NODE.to_string(), ActivationFn::Tanh, LABEL_SIZE)
        .add_edge(LABEL_NODE.to_string(), SENSOR_NODE.to_string())
        .build();

    pcn.randomize_weights_uniform(&mut rng);

    let initial_error = test_it(&SENSOR_NODE.to_string(), &LABEL_NODE.to_string(), &mut pcn);

    learn_it(
        &SENSOR_NODE.to_string(),
        &LABEL_NODE.to_string(),
        &mut pcn,
        &mut rng,
        LEARNING_STEPS,
    );

    let final_error = test_it(&SENSOR_NODE.to_string(), &LABEL_NODE.to_string(), &mut pcn);

    println!(
        "initial error: {} => final error: {}",
        initial_error, final_error
    );

    println!(
        "Improvement: {} (should be > 0)",
        initial_error - final_error
    );

    // println!("Final network: {pcn:#?}");
}
