mod tensor;

fn main() {
    let tensor_a = tensor::Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let tensor_b = tensor::Tensor::new(vec![2.0, 1.0, 7.0, 8.0], vec![2, 2]);
    let tensor_c: tensor::Tensor = tensor::Tensor::add(&tensor_a, &tensor_b);
    println!("{:?}", tensor_c);
}
