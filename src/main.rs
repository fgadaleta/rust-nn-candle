use candle_core::{
    cuda_backend::cudarc::driver::sys::CU_TRSF_READ_AS_INTEGER, DType, Device, Result, Tensor, D,
};
use candle_nn::{self, loss, ops, Dropout, Linear, Module, Optimizer, VarBuilder, VarMap, SGD};

const IN_DIMS: usize = 2;
const OUT_DIMS: usize = 1;
const LEARNING_RATE: f64 = 0.0003;
const EPOCHS: usize = 5;

#[derive(Debug)]
struct MyNeuralNetwork {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
}

impl MyNeuralNetwork {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        /*
           INPUT                          OUTPUT
           LAYER1          LAYER2         LAYER3
           2x10   ----->   10x10  ----->  10x1
        */

        // TODO don't just fail miserably (unwrap)
        let layer1 = candle_nn::linear(IN_DIMS, 10, vb.pp("l1")).unwrap();
        let layer2 = candle_nn::linear(10, 10, vb.pp("l2")).unwrap();
        let layer3 = candle_nn::linear(10, OUT_DIMS, vb.pp("l3")).unwrap();

        Ok(MyNeuralNetwork {
            layer1,
            layer2,
            layer3,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.layer1.forward(&x).unwrap();
        let x = x.relu().unwrap();
        let x = self.layer2.forward(&x).unwrap();
        let x = x.relu().unwrap();
        let x = self.layer3.forward(&x).unwrap();
        let x = x.relu().unwrap();
        Ok(x)
    }
}

// pub struct Dataset {
//     train_data:
// }

fn main() {
    let device = Device::cuda_if_available(0);
    dbg!(&device);
    let device = match device {
        Ok(d) => d,
        Err(_) => return,
    };

    // Irrelevant (just examples)
    // let a = Tensor::randn(0f32, 1., (2, 3), &device).unwrap();
    // let b = Tensor::randn(0f32, 1., (3, 4), &device).unwrap();
    // let c = a.matmul(&b).unwrap();
    // let res_dims = c.dims();
    // dbg!(&res_dims);
    // println!("{c}");

    let var_map = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let model = MyNeuralNetwork::new(var_builder).unwrap();
    dbg!(&model);
    let all_vars = var_map.all_vars();
    let mut sgd = candle_nn::SGD::new(all_vars, LEARNING_RATE).unwrap();

    // Create training data
    let training_data_x: Vec<u32> = vec![0, 2, 2, 4, 4, 6, 4, 8, 11, 9, 13, 87, 57, 69, 43, 31];
    let training_data_y: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let training_data_tensor_x = Tensor::from_vec(
        training_data_x.clone(),
        (training_data_x.len() / 2, 2),
        &device,
    )
    .unwrap()
    .to_dtype(DType::F32)
    .unwrap();
    let training_data_tensor_y =
        Tensor::from_vec(training_data_y.clone(), training_data_y.len(), &device)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap();

    // Create testing data
    let test_data_x: Vec<u32> = vec![51, 77, 39, 61, 50, 60, 54, 84];
    let test_data_y: Vec<u32> = vec![1, 1, 0, 0];
    let test_data_tensor_x =
        Tensor::from_vec(test_data_x.clone(), (test_data_x.len() / 2, 2), &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
    let test_data_tensor_y = Tensor::from_vec(test_data_y.clone(), test_data_y.len(), &device)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap();

    // Training
    for epoch in 1..EPOCHS {
        println!("Training epoch {epoch}");
        // TODO
        let training_pairs = training_data_tensor_x.to_device(&device).unwrap();
        let training_labels = training_data_tensor_y.to_device(&device).unwrap();
        let res = model.forward(&training_pairs).unwrap();
        let log_sm = ops::log_softmax(&res, D::Minus1).unwrap();
        let loss = loss::nll(&log_sm, &training_labels).unwrap();
        let _ = sgd.backward_step(&loss);
        println!("Loss: {loss}");
    }

    let test_data_tensor_x = test_data_tensor_x.to_device(&device).unwrap();
    let test_data_tensor_y = test_data_tensor_y.to_device(&device).unwrap();

    let pred = model.forward(&test_data_tensor_x).unwrap();
    let pred = pred
        .argmax(D::Minus1)
        .unwrap()
        .eq(&test_data_tensor_y)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<u32>()
        .unwrap();
    println!("Predicted: {pred}");
}
