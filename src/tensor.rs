use std::{fmt, path::Iter};

#[derive(PartialEq, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub grad: Option<Box<Tensor>>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(shape.iter().copied().reduce(|a,b| a*b).unwrap(), data.len());
        Tensor {
            data: data,
            shape: shape,
            grad: None
        }
    }

    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Transpose only supports 2D tensors.");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut transposed_data = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                transposed_data[j * rows + i] = self.data[i * cols + j];
            }
        }
        Tensor {
            data: transposed_data,
            shape: vec![cols, rows],
            grad: None,
        }
    }
    pub fn add(x: &Tensor, y: &Tensor) -> Tensor {
        assert_eq!(x.shape, y.shape);
        Tensor {
            data: x.data.iter().zip(y.data.iter()).map(|(a,b)| a+b).collect(),
            shape: x.shape.clone(),
            grad: None
        }
    }
    pub fn matmul(x: &Tensor, y: &Tensor) -> Tensor {
        // Only support 2D mul for now
        assert_eq!(x.shape.len(), 2);
        assert_eq!(y.shape.len(), 2);
        let (m, n) = (x.shape[0], x.shape[1]);
        let (p, q) = (y.shape[0], y.shape[1]);

        assert_eq!(n, p);

        let mut result_data = vec![0.0; m * q];

        for i in 0..m {
            for j in 0..q {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += x.data[i * n + k] * y.data[k * q + j];
                }
                result_data[i * q + j] = sum;
            }
        }

        Tensor {
            data: result_data,
            shape: vec![m, q],
            grad: None,
        }
    }
}
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor{:?}: {:?}", self.shape, self.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tensor_creation_correct() {
        let tensor_a: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]); 
    }
    #[test]
    fn test_tensor_creation_shape_mismatch() {
        let result = std::panic::catch_unwind(|| {
            let tensor_a: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![3, 2]); 
        });
        assert!(result.is_err(), "Tensor shape does not match number of data elements.");
    }
    #[test]
    fn test_tensor_addition_correct() {
        let tensor_a: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let tensor_b: Tensor = Tensor::new(vec![2.0, 1.0, -7.0, 8.0], vec![2, 2]);
        let result: Tensor = Tensor::add(&tensor_a,&tensor_b);
        let expected_result = Tensor::new(vec![3.0, 3.0, -4.0, 12.0], vec![2, 2]);
        assert_eq!(expected_result, result);
    }
    #[test]
    fn test_tensor_addition_shape_mismatch() {
        let tensor_a: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let tensor_b: Tensor = Tensor::new(vec![2.0, 1.0, -7.0, 8.0], vec![4, 1]);
        let result = std::panic::catch_unwind(|| {
            Tensor::add(&tensor_a, &tensor_b);
        });
        assert!(result.is_err(), "addition panic due to shape mismatch");
    }
    #[test]
    fn test_tensor_matmul_correct() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = Tensor::matmul(&a, &b);
        let expected = Tensor::new(vec![19.0, 22.0, 43.0, 50.0], vec![2, 2]);
        assert_eq!(result, expected);
    }
    #[test]
    fn test_tensor_matmul_shape_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0], vec![3, 1]);

        let result = std::panic::catch_unwind(|| {
            Tensor::matmul(&a, &b);
        });

        assert!(result.is_err(), "matmul panic due to shape mismatch");
    }

}