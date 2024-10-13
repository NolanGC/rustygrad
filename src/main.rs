mod tensor;
mod optim;
mod nn;

use nn::{Linear, ReLU, Sequential, Module};
use tensor::Tensor;
use optim::Optim;

fn main() {
    let linear1 = Box::new(Linear::new(784, 128, true));
    let relu1 = Box::new(ReLU::new());
    let linear2 = Box::new(Linear::new(128, 10, true));
    let mut model = Sequential::new(vec![linear1, relu1, linear2]);

    let input_data = vec![0.5; 784]; // Example data
    let input = Tensor::new(input_data, vec![784, 1]);
    let output = model.forward(&input);
    println!("Output Tensor: {:?}", output);
    let loss_grad_data = vec![0.1; 10];
    let loss_grad = Tensor::new(loss_grad_data, vec![10, 1]);
    model.backward(&loss_grad);
    let mut params = model.parameters_mut();
    let mut optimizer = Optim::new(params, 0.01);
    optimizer.update();
    optimizer.zero_grad();
    println!("Parameters updated successfully.");
}