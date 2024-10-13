use crate::tensor::Tensor;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

pub trait Module {
    fn forward(&mut self, x: &Tensor) -> Tensor;
    fn backward(&mut self, j_out: &Tensor) -> Tensor;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
}

pub struct Linear {
    in_features: usize,
    out_features: usize,
    x_in: Option<Tensor>,
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let normal: Normal<f32> = Normal::new(0.0, 1.0).unwrap();
        let shape: Vec<usize> = vec![in_features, out_features];
        let mut data: Vec<f32> = Vec::with_capacity(in_features * out_features);
        let mut rng = thread_rng();

        for _ in 0..(in_features * out_features) {
            data.push(normal.sample(&mut rng));
        }

        let weight_tensor = Tensor::new(data, shape);

        let bias_tensor = if bias {
            let bias_data = vec![0.0; out_features];
            Some(Tensor::new(bias_data, vec![out_features, 1]))
        } else {
            None
        };

        Linear {
            in_features,
            out_features,
            x_in: None,
            weight: weight_tensor,
            bias: bias_tensor,
        }
    }
}

impl Module for Linear {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        assert_eq!(
            x.shape[0],
            self.in_features,
            "Input tensor shape mismatch."
        );
        let mut product = Tensor::matmul(&self.weight.transpose(), x);
        if let Some(bias_tensor) = &self.bias {
            product = Tensor::add(&product, bias_tensor);
        }
        self.x_in = Some(x.clone());
        product
    }
    fn backward(&mut self, j_out: &Tensor) -> Tensor {
        if let Some(x_in) = &self.x_in {
            let j_out_transposed = j_out.transpose();
            let djdw = Tensor::matmul(x_in, &j_out_transposed);
            self.weight.grad = Some(Box::new(djdw));
        } else {
            panic!("No cached inputs during backward pass");
        }
        let j_in = Tensor::matmul(&self.weight, j_out);

        if let Some(bias) = &mut self.bias {
            bias.grad = Some(Box::new(j_out.clone()));
        }
        j_in
    }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
}
pub struct ReLU {
    x_in: Option<Tensor>,
}

impl ReLU {
    pub fn new() -> Self {
        ReLU { x_in: None }
    }
}

impl Module for ReLU {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        self.x_in = Some(x.clone());
        let data = x.data.iter().map(|&v| v.max(0.0)).collect();
        Tensor::new(data, x.shape.clone())
    }

    fn backward(&mut self, j_out: &Tensor) -> Tensor {
        if let Some(ref x_in) = self.x_in {
            let grad = x_in
                .data
                .iter()
                .zip(j_out.data.iter())
                .map(|(&x, &j)| if x > 0.0 { j } else { 0.0 })
                .collect();
            Tensor::new(grad, j_out.shape.clone())
        } else {
            panic!("No cached inputs during backward pass");
        }
    }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
    }
}

impl Module for Sequential {
    fn forward(&mut self, inp: &Tensor) -> Tensor {
        let mut x = inp.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x);
        }
        x
    }
    fn backward(&mut self, j_out: &Tensor) -> Tensor {
        let mut grad = j_out.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
        grad
    }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        params
    }
}

