mod tensor;
mod nn;

use nn::Linear;
use tensor::Tensor;

fn main() {
    let mut linear_layer = Linear::new(2, 2, true);
    let input_tensor = Tensor::new(vec![1.0, 0.0], vec![2, 1]);
    let output_tensor = linear_layer.forward(input_tensor);
    println!("{:?}", output_tensor);
}
