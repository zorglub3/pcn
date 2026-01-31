use pcn::mikko::*;

const INFERENCE_STEPS: usize = 512;

fn main() {
    let mut rng = rand::rng();

    let mut pcn = Network::new(&[2, 1000, 1500, 1500, 1]);

    pcn.randomize_weights(&mut rng);

    pcn.reset(&mut rng);
    pcn.set_input(&[1., 1.]);

    for _i in 0..INFERENCE_STEPS {
        pcn.inference(0.2);
    }
}
