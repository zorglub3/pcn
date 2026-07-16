const F64_TRUE: f64 = 1.;
const F64_FALSE: f64 = -1.;

pub fn bool_to_f64<const C: usize>(pattern: &[bool; C]) -> [f64; C] {
    let mut x = [0.; C];
    for i in 0..C {
        x[i] = if pattern[i] { F64_TRUE } else { F64_FALSE };
    }
    x
}

pub fn square_error(pattern1: &[f64], pattern2: &[f64]) -> f64 {
    assert_eq!(pattern1.len(), pattern2.len());

    let mut acc = 0.;

    for i in 0..pattern1.len() {
        let e = pattern1[i] - pattern2[i];
        acc += e * e;
    }

    acc
}
