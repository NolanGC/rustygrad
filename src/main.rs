mod nn;
mod tensor;
mod optim;
mod function;

use nn::{Linear, ReLU, Sequential, Module};
use tensor::Tensor;
use optim::Optim;
use function::sigmoid;

fn main() {
    let linear1 = Box::new(Linear::new(2, 2, true));
    let relu1 = Box::new(ReLU::new());
    let linear2 = Box::new(Linear::new(2, 1, true));
    let mut model = Sequential::new(vec![linear1, relu1, linear2]);
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    let learning_rate = 0.1;

    let epochs = 100;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (input_vals, target_vals) in &training_data {
            let input = Tensor::new(input_vals.clone(), vec![2, 1]);
            let output = model.forward(&input);
            let target = Tensor::new(target_vals.clone(), vec![1, 1]);
            let loss = compute_mse_loss(&output, &target);
            total_loss += loss.data[0];
            let grad_loss = compute_mse_loss_grad(&output, &target);
            //println!("{:?}", grad_loss);
            model.backward(&grad_loss);
            {
                let mut params = model.parameters_mut();
                let mut optimizer = Optim::new(params, learning_rate);
                optimizer.update();
                optimizer.zero_grad();
            }
        }
        if epoch % 10 == 0 {
            println!("Epoch {:5}: Loss = {:.4}", epoch, total_loss / 4.0);
        }
    }

    println!("\nTraining completed.\nTesting the network on XOR inputs:");
    for (input_vals, target_vals) in &training_data {
        let input = Tensor::new(input_vals.clone(), vec![2, 1]);
        let output = model.forward(&input);
        let predicted = sigmoid(output.data[0]);
        println!(
            "Input: {:?}, Predicted: {:.4}, Target: {}",
            input_vals, predicted, target_vals[0]
        );
    }
}

fn compute_mse_loss(output: &Tensor, target: &Tensor) -> Tensor {
    assert!(output.shape == target.shape, "Output and target shapes must match.");

    let mse_data: Vec<f32> = output
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(&o, &t)| (o - t).powi(2))
        .collect();

    let loss = mse_data.iter().sum::<f32>() / output.data.len() as f32;

    Tensor::new(vec![loss], vec![1, 1])
}


fn compute_mse_loss_grad(output: &Tensor, target: &Tensor) -> Tensor {
    assert!(output.shape == target.shape, "Output and target shapes must match.");

    let grad_data: Vec<f32> = output
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(&o, &t)| 2.0 * (o - t) / output.data.len() as f32)
        .collect();

    Tensor::new(grad_data, output.shape.clone())
}
