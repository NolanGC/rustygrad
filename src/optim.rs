
use crate::tensor::Tensor;

pub struct Optim<'a> {
    params: Vec<&'a mut Tensor>,
    lr: f32,
}

impl<'a> Optim<'a> {
    pub fn new(params: Vec<&'a mut Tensor>, lr: f32) -> Self {
        Optim { params, lr }
    }

    pub fn zero_grad(&mut self) {
        for p in self.params.iter_mut() {
            if let Some(ref mut grad_tensor) = p.grad {
                for g in grad_tensor.data.iter_mut() {
                    *g = 0.0;
                }
            }
        }
    }

    pub fn update(&mut self) {
        for p in self.params.iter_mut() {
            if let Some(ref grad_tensor) = p.grad {
                for (i, g) in grad_tensor.data.iter().enumerate() {
                    p.data[i] -= self.lr * g;
                }
            }
        }
    }
}