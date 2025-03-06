use std::fmt::Debug;
use std::ops::{AddAssign, Index, IndexMut, Mul};

pub struct DMatrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Debug> DMatrix<T> {
    #[allow(dead_code)]
    pub fn pp(&self) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                print!("{:?} ", self[(r, c)]);
            }
            println!("");
        }
    }
}

impl<T> DMatrix<T> {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    fn row_col_to_index(&self, r: usize, c: usize) -> usize {
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
    pub fn add_vecs_mul(&mut self, alpha: T, v1: &[T], v2: &[T]) {
        debug_assert_eq!(self.rows, v1.len());
        debug_assert_eq!(self.cols, v2.len());

        for r in 0..self.rows() {
            for c in 0..self.cols() {
                self[(r, c)] += alpha * v1[r] * v2[c];
            }
        }
    }

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

    pub fn mul_row_vec(&self, v: &[T], r: usize) -> T {
        debug_assert_eq!(self.cols, v.len());
        debug_assert!(r < self.rows);

        let mut acc = T::default();

        let mut i1 = r * self.cols;

        for i2 in 0..v.len() {
            acc += self.data[i1] * v[i2];
            i1 += 1;
        }

        acc
    }

    pub fn mul_col_vec(&self, v: &[T], c: usize) -> T {
        debug_assert_eq!(self.rows, v.len());
        debug_assert!(c < self.cols);

        let mut acc = T::default();
        let mut i1 = c;

        for i2 in 0..v.len() {
            acc += self.data[i1] * v[i2];
            i1 += self.cols;
        }

        acc
    }

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

    pub fn mul_vec(&self, input: &[T], output: &mut [T]) {
        debug_assert_eq!(self.cols, input.len());
        debug_assert_eq!(self.rows, output.len());

        for i in 0..output.len() {
            output[i] = self.mul_row_vec(input, i);
        }
    }

    pub fn mul_vec_add(&self, input: &[T], output: &mut [T]) {
        debug_assert_eq!(self.cols, input.len());
        debug_assert_eq!(self.rows, output.len());

        for i in 0..output.len() {
            output[i] += self.mul_row_vec(input, i);
        }
    }

    pub fn trans_mul_vec(&self, input: &[T], output: &mut [T]) {
        debug_assert_eq!(self.rows, input.len());
        debug_assert_eq!(self.cols, output.len());

        for i in 0..output.len() {
            output[i] = self.mul_col_vec(input, i);
        }
    }
}

impl<T: AddAssign + Copy> DMatrix<T> {
    pub fn add_matrix(&mut self, mat: &DMatrix<T>) {
        debug_assert_eq!(self.rows, mat.rows);
        debug_assert_eq!(self.cols, mat.cols);

        for i in 0..self.data.len() {
            self.data[i] += mat.data[i];
        }
    }
}

impl<T: Clone> DMatrix<T> {
    pub fn assign(&mut self, m: &DMatrix<T>) {
        debug_assert_eq!(self.rows, m.rows);
        debug_assert_eq!(self.cols, m.cols);

        for i in 0..self.data.len() {
            self.data[i] = m.data[i].clone();
        }
    }

    pub fn add_row(&mut self, row: &[T]) {
        debug_assert_eq!(self.cols, row.len());

        for v in row {
            self.data.push(v.clone());
        }

        self.rows += 1;
    }

    pub fn new(rows: usize, cols: usize, default: T) -> Self {
        let data = vec![default; rows * cols];

        Self { rows, cols, data }
    }
}
