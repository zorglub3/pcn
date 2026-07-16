use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::{AddAssign, Index, IndexMut, Mul, Range};

/// A Dense matrix with data stored row-wise
#[derive(Serialize, Deserialize, Clone)]
pub struct DMatrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Debug> DMatrix<T> {
    /// Prettyh print the matrix to stdout.
    #[allow(dead_code)]
    pub fn pp(&self) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                print!("{:?} ", self[(r, c)]);
            }
            println!();
        }
    }
}

impl<T> DMatrix<T> {
    /// Get the number of rows of the matrix
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get an iterator over the rows
    pub fn rows_range(&self) -> Range<usize> {
        0..self.rows
    }

    /// Get the number of columns for the matrix
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get an iterator (a range really) over columns in the matrix
    pub fn cols_range(&self) -> Range<usize> {
        0..self.cols
    }

    /// Compute the index in the data block for some row and column
    #[inline]
    fn row_col_to_index(&self, r: usize, c: usize) -> usize {
        debug_assert!(r < self.rows());
        debug_assert!(c < self.cols());
        r * self.cols + c
    }
}

impl<T> Index<(usize, usize)> for DMatrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &T {
        let idx = self.row_col_to_index(index.0, index.1);
        &self.data[idx]
    }
}

impl<T> IndexMut<(usize, usize)> for DMatrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        let idx = self.row_col_to_index(index.0, index.1);
        &mut self.data[idx]
    }
}

impl<T: Mul<Output = T> + Default + AddAssign + Copy> DMatrix<T> {
    /// Given two column vectors, v1 and v2, compute the matrix v1 * v2^T.
    /// Then scale the matrix by alpha and add it to self. Note that
    /// |v1| == rows(self) and |v2| == columns(self)
    #[allow(unused)]
    pub fn add_vecs_mul(&mut self, alpha: T, v1: &[T], v2: &[T]) {
        debug_assert_eq!(self.rows, v1.len());
        debug_assert_eq!(self.cols, v2.len());

        for r in 0..self.rows() {
            for c in 0..self.cols() {
                self[(r, c)] += alpha * v1[r] * v2[c];
            }
        }
    }

    /// Multiply one row of 'self' with one column of 'm'.
    #[allow(dead_code)]
    pub fn mul_row_col(&self, m: &DMatrix<T>, r: usize, c: usize) -> T {
        debug_assert_eq!(self.cols, m.rows);

        let mut acc = T::default();

        let mut i1 = r * self.cols;
        let mut i2 = c;

        for _i in 0..self.cols {
            acc += self.data[i1] * m.data[i2];
            i1 += 1;
            i2 += m.cols;
        }

        acc
    }

    /// Multiply one row of self with vector 'v' and accumulate
    /// the result.
    pub fn mul_row_vec(&self, v: &[T], r: usize) -> T {
        debug_assert_eq!(self.cols, v.len());
        debug_assert!(r < self.rows);

        let mut acc = T::default();

        let mut i1 = r * self.cols;

        for item in v {
            acc += self.data[i1] * *item;
            i1 += 1;
        }

        acc
    }

    /// Multiply one column of self with vector 'v' and accumulate
    /// the result.
    pub fn mul_col_vec(&self, v: &[T], c: usize) -> T {
        debug_assert_eq!(self.rows, v.len());
        debug_assert!(c < self.cols);

        let mut acc = T::default();
        let mut i1 = c;

        for item in v {
            acc += self.data[i1] * *item;
            i1 += self.cols;
        }

        acc
    }

    /// Multiply matrix m1 with matrix m2 and assign the result to self.
    #[allow(dead_code)]
    pub fn mul_assign(&mut self, m1: &DMatrix<T>, m2: &DMatrix<T>) {
        debug_assert_eq!(self.rows, m1.rows);
        debug_assert_eq!(self.cols, m2.cols);
        debug_assert_eq!(m1.cols, m2.rows);

        for r in 0..self.rows {
            for c in 0..self.cols {
                self[(r, c)] = m1.mul_row_col(m2, r, c);
            }
        }
    }

    /// Multiply self with input vector (a column vector) to produce
    /// output (overwritten).
    #[allow(dead_code)]
    pub fn mul_vec(&self, input: &[T], output: &mut [T]) {
        debug_assert_eq!(self.cols, input.len());
        debug_assert_eq!(self.rows, output.len());

        for (i, item) in output.iter_mut().enumerate() {
            *item = self.mul_row_vec(input, i);
        }
    }

    /// Multiply self with input column vector and add the result to
    /// output vector.
    pub fn mul_vec_add(&self, input: &[T], output: &mut [T]) {
        debug_assert_eq!(self.cols, input.len());
        debug_assert_eq!(self.rows, output.len());

        for (i, item) in output.iter_mut().enumerate() {
            *item += self.mul_row_vec(input, i);
        }
    }

    /// Transpose self and multiply with column vector and add the
    /// result to output vector.
    pub fn trans_mul_vec_add(&self, input: &[T], output: &mut [T]) {
        debug_assert_eq!(self.rows, input.len());
        debug_assert_eq!(self.cols, output.len());

        for (i, item) in output.iter_mut().enumerate() {
            *item += self.mul_col_vec(input, i);
        }
    }
}

impl<T: AddAssign + Copy> DMatrix<T> {
    /// Add matrix mat to self.
    #[allow(dead_code)]
    pub fn add_matrix(&mut self, mat: &DMatrix<T>) {
        debug_assert_eq!(self.rows, mat.rows);
        debug_assert_eq!(self.cols, mat.cols);

        for i in 0..self.data.len() {
            self.data[i] += mat.data[i];
        }
    }
}

impl<T: Clone> DMatrix<T> {
    /// Assign values in matrix m to self - overwrite old values.
    #[allow(dead_code)]
    pub fn assign(&mut self, m: &DMatrix<T>) {
        debug_assert_eq!(self.rows, m.rows);
        debug_assert_eq!(self.cols, m.cols);

        for i in 0..self.data.len() {
            self.data[i] = m.data[i].clone();
        }
    }

    /// Add one row to self, thus incrementing row count by one,
    /// Note: Does not work for empty matrix (with no rows).
    #[allow(dead_code)]
    pub fn add_row(&mut self, row: &[T]) {
        debug_assert_eq!(self.cols, row.len());

        for v in row {
            self.data.push(v.clone());
        }

        self.rows += 1;
    }

    /// Create a new matrix with given number of rows and columns.
    /// Clone the default value.
    pub fn new(rows: usize, cols: usize, default: T) -> Self {
        let data = vec![default; rows * cols];

        Self { rows, cols, data }
    }
}

impl DMatrix<f64> {
    /// Randomize the values in matrix. Assign a new random value to
    /// each element. The values are uniformly distributed between
    /// minus amount and plus amount (not incl).
    pub fn randomize(&mut self, amount: f64, rng: &mut impl Rng) {
        if amount <= f64::EPSILON {
            self.data.fill(0.);
        } else {
            for item in &mut self.data {
                *item = rng.random_range(-amount..amount);
            }
        }
    }

    pub fn randomize_xavier(&mut self, rng: &mut impl Rng) {
        let amount = (6. / (self.rows() + self.cols()) as f64).sqrt();
        for item in &mut self.data {
            *item = rng.random_range(-amount..amount);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn new_matrix_has_correct_rows_and_cols() {
        let m = DMatrix::new(10, 20, 1_i32);
        assert_eq!(m.rows(), 10);
        assert_eq!(m.cols(), 20);
    }

    #[test]
    fn matrix_computes_correct_index() {
        let m = DMatrix::new(3, 5, 1_i32);
        assert_eq!(m.row_col_to_index(0, 2), 2);
        assert_eq!(m.row_col_to_index(2, 0), 2 * 5);
        assert_eq!(m.row_col_to_index(1, 1), 1 * 5 + 1);
    }

    fn test_matrix_2_3() -> DMatrix<f64> {
        let mut m = DMatrix::new(2, 3, 0_f64);

        for r in 0..2 {
            for c in 0..3 {
                m[(r, c)] = (r as f64) + (c as f64) * 10.;
            }
        }

        m
    }

    fn test_matrix_3_2() -> DMatrix<f64> {
        let mut m = DMatrix::new(3, 2, 0_f64);

        for r in 0..3 {
            for c in 0..2 {
                m[(r, c)] = (r as f64) * 2. + (c as f64) * 5.;
            }
        }

        m
    }

    #[test]
    fn matrix_index() {
        let m = test_matrix_2_3();

        assert_eq!(m[(1, 1)], 11.);
        assert_eq!(m[(1, 2)], 21.);
        assert_eq!(m[(0, 2)], 20.);
    }

    #[test]
    fn test_add_vecs_mul() {
        let mut m = test_matrix_2_3();

        let v1 = [2., 3.];
        let v2 = [1., 2., 3.];

        m.add_vecs_mul(0.5, &v1, &v2);

        assert_eq!(m[(1, 1)], 11. + 0.5 * 3. * 2.);
    }

    #[test]
    fn test_mul_row_col() {
        let m = test_matrix_2_3();
        let m2 = test_matrix_3_2();

        // m(r=1):   1, 11, 21
        // m2(c=0):  0,  2,  4
        assert_eq!(m.mul_row_col(&m2, 1, 0), 22. + 84.);
    }

    #[test]
    fn test_mul_row_vec() {
        let m = test_matrix_2_3();
        let v = [-1., -2., 3.];

        assert_eq!(m.mul_row_vec(&v, 1), -1. + -22. + 63.);
    }

    #[test]
    fn test_mul_col_vec() {
        let m = test_matrix_2_3();
        let v = [-1., 1.];

        assert_eq!(m.mul_col_vec(&v, 1), -10. + 11.);
    }

    #[test]
    fn test_mul_vec() {
        let m = test_matrix_2_3();
        let v = [1., 2., 3.];
        let mut o = [0.; 2];

        m.mul_vec(&v, &mut o);

        assert_eq!(o[0], 2. * 10. + 3. * 20.);
        assert_eq!(o[1], 1. * 1. + 2. * 11. + 3. * 21.);
    }

    #[test]
    fn test_mul_vec_add() {
        let m = test_matrix_2_3();
        let v = [1., 2., 3.];
        let mut o = [-1., -2.];

        m.mul_vec_add(&v, &mut o);

        assert_eq!(o[0], -1. + 2. * 10. + 3. * 20.);
        assert_eq!(o[1], -2. + 1. * 1. + 2. * 11. + 3. * 21.);
    }

    #[test]
    fn test_trans_mul_vec_add() {
        let m = test_matrix_2_3();
        let v = [-1., 2.];
        let mut o = [1., 2., 3.];

        m.trans_mul_vec_add(&v, &mut o);

        assert_eq!(o[0], 1. + 2.);
        assert_eq!(o[1], 2. + -1. * 10. + 2. * 11.);
        assert_eq!(o[2], 3. + -1. * 20. + 2. * 21.);
    }
}
