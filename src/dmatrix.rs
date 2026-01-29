use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::{AddAssign, Index, IndexMut, Mul};

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

    /// Get the number of columns for the matrix
    pub fn cols(&self) -> usize {
        self.cols
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
}
