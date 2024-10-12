use crate::tensor::Tensor;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

pub struct Linear {
    in_features: usize,
    out_features: usize,
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let normal: Normal<f32> = Normal::new(0.0, 1.0).unwrap();
        let shape: Vec<usize> = vec![in_features, out_features];
        let mut data: Vec<f32> = Vec::with_capacity(in_features * out_features);
        let mut rng: rand::prelude::ThreadRng = thread_rng();

        for _ in 0..(in_features * out_features) {
            data.push(normal.sample(&mut rng));
        }

        let weight_tensor: Tensor = Tensor::new(data, shape);

        let bias_tensor: Option<Tensor> = if bias {
            let bias_data: Vec<f32> = vec![0.0; out_features];
            Some(Tensor::new(bias_data, vec![out_features, 1]))
        } else {
            None
        };

        Linear {
            in_features,
            out_features,
            weight: weight_tensor,
            bias: bias_tensor,
        }
    }
    pub fn forward(&mut self, x: Tensor) -> Tensor {
        assert_eq!(
            x.shape()[0],
            self.in_features,
            "Input tensor shape mismatch."
        );
        let mut product = Tensor::matmul(&self.weight.transpose(), &x);
        if let Some(bias_tensor) = &self.bias {
            product = Tensor::add(&product, bias_tensor);
        }
        product
    }
    pub fn backward(&mut self, g_out: &Tensor) -> Tensor {
        let g_in : Tensor = Tensor::matmul(&g_out, &self.weight.transpose());
        g_in 
    }
}
