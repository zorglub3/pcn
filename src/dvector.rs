#[inline]
pub fn hadamard_inplace(a: &[f64], b: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());

    for i in 0..a.len() {
        b[i] *= a[i];
    }
}

#[allow(dead_code)]
#[inline]
pub fn add_inplace(a: &[f64], b: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());

    for i in 0..a.len() {
        b[i] += a[i];
    }
}

#[inline]
pub fn scale_add_inplace(gamma: f64, a: &[f64], b: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());

    for i in 0..a.len() {
        b[i] += a[i] * gamma;
    }
}

#[inline]
pub fn sub_inplace(a: &[f64], b: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());

    for i in 0..a.len() {
        b[i] -= a[i];
    }
}

#[allow(dead_code)]
#[inline]
pub fn scale_sub_inplace(gamma: f64, a: &[f64], b: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());

    for i in 0..a.len() {
        b[i] -= a[i] * gamma;
    }
}

#[inline]
pub fn randomize_vec(amount: f64, v: &mut [f64], rng: &mut impl rand::Rng) {
    for item in v {
        *item = rng.random_range(-amount..amount);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_zero_length() {
        let a = [];
        let mut b = [];

        hadamard_inplace(&a, &mut b);

        assert_eq!(b.len(), 0);
    }

    #[test]
    fn test_product() {
        let a = [-1., 2., 3.];
        let mut b = [2., 3., 4.];

        hadamard_inplace(&a, &mut b);

        assert_eq!(b[0], -2.);
        assert_eq!(b[1], 6.);
        assert_eq!(b[2], 12.);
    }
}
