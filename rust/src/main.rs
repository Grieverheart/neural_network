use std::path::Path;
use std::convert::AsRef;
use std::io::Read;

use rand::Rng;
use rand::prelude::thread_rng;
use rand::distributions::StandardNormal;
use rand::seq::SliceRandom;

struct NnLinearLayer
{
    input_size:  usize,
    output_size: usize,

    weights: Vec<f64>,
    biases:  Vec<f64>,
}

impl NnLinearLayer
{
    fn new(input_size: usize, output_size: usize) -> NnLinearLayer
    {
        let weights = thread_rng().sample_iter(&StandardNormal)
                        .take(input_size * output_size).collect();
        let biases  = thread_rng().sample_iter(&StandardNormal)
                        .take(output_size).collect();

        NnLinearLayer{
            input_size: input_size,
            output_size: output_size,
            weights: weights, biases: biases
        }
    }

    fn process(&self, input: &[f64]) -> Vec<f64>
    {
        assert!(input.len() == self.input_size);
        let mut output = self.biases.clone();
        for i in 0..self.output_size
        {
            for j in 0..self.input_size
            {
                output[i] += self.weights[i * self.input_size + j] * input[j];
            }
        }
        output
    }
}

fn sigmoid(x: f64) -> f64
{
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_prime(x: f64) -> f64
{
    let s = sigmoid(x);
    s * (1.0 - s)
}

fn nn_run(layers: &[NnLinearLayer], input: &[f64]) -> Vec<f64>
{
    let mut output = vec![0.0; input.len()];
    output.clone_from_slice(input);

    for i in 0..layers.len()
    {
        output = layers[i].process(&output).iter().map(|&x| sigmoid(x)).collect();
    }
    output
}

fn nn_backprop(layers: &[NnLinearLayer], input: &[f64], expected_output: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>)
{
    let num_layers = layers.len();

    // Feedforward
    let mut activation     = vec![Vec::new(); num_layers + 1];
    let mut weighted_input = vec![Vec::new(); num_layers];

    activation[0] = input.to_vec();

    for i in 0..num_layers
    {
        let wi: Vec<f64> = layers[i].process(&activation[i]);
        let layer_activation = wi.iter().map(|&x| sigmoid(x)).collect::<Vec<f64>>();

        weighted_input[i] = wi.to_vec();
        activation[i + 1] = layer_activation.to_vec();
    }

    let mut error = ((0..activation[num_layers].len()).map(
        |i| (activation[num_layers][i] - expected_output[i]) *
            sigmoid_prime(weighted_input[num_layers - 1][i])
    ).collect::<Vec<f64>>()).to_vec();

    let mut delta_weights = vec![Vec::new(); num_layers];
    let mut delta_biases  = vec![Vec::new(); num_layers];

    for lidx in (0..num_layers).rev()
    {
        let layer = &layers[lidx];
        delta_weights[lidx].resize_with(layer.input_size * layer.output_size, || 0.0);
        delta_biases[lidx].resize_with(layer.input_size, || 0.0);

        for k in 0..layer.output_size
        {
            for j in 0..layer.input_size
            {
                delta_weights[lidx][k * layer.input_size + j] = error[k] * activation[lidx][j];
            }
            delta_biases[lidx][k] = error[k];
        }

        if lidx == 0
        {
            break;
        }

        let mut new_error = vec![0.0; layer.input_size];

        for j in 0..layer.input_size
        {
            for k in 0..layer.output_size
            {
                new_error[j] += layer.weights[k * layer.input_size + j] * error[k];
            }
            new_error[j] *= sigmoid_prime(weighted_input[lidx - 1][j]);
        }

        error = new_error;
    }

    (delta_weights, delta_biases)
}

fn sgd_train(
    layers: &mut [NnLinearLayer],
    train_data: &Vec<Vec<f64>>, train_labels: &Vec<Vec<f64>>,
    test_data:  &Vec<Vec<f64>>, test_labels:  &Vec<Vec<f64>>,
    num_epochs: usize, batch_size: usize, eta: f64
){
    for ei in 0..num_epochs
    {
        println!("    Training epoch {}", ei);
        let timer = std::time::Instant::now();
        let mut index_batches = (1..train_labels.len()).collect::<Vec<_>>();
        index_batches.shuffle(&mut thread_rng());

        let index_batches = index_batches.chunks(batch_size);
        for batch_indices in index_batches
        {
            let mut batch_delta_weights: Vec<Vec<f64>> = vec![Vec::new(); layers.len()];
            let mut batch_delta_biases: Vec<Vec<f64>>  = vec![Vec::new(); layers.len()];

            for lidx in 0..layers.len()
            {
                batch_delta_weights[lidx].resize_with(layers[lidx].weights.len(), Default::default);
                batch_delta_biases[lidx].resize_with(layers[lidx].biases.len(), Default::default);
            }

            for &sample_index in batch_indices
            {
                let actual_batch_size = batch_indices.len();
                let (delta_weights, delta_biases) = nn_backprop(layers, &train_data[sample_index], &train_labels[sample_index]);

                // Accumulate
                for lidx in 0..layers.len()
                {
                    for i in 0..layers[lidx].output_size
                    {
                        for j in 0..layers[lidx].input_size
                        {
                            batch_delta_weights[lidx][i * layers[lidx].input_size + j] +=
                                delta_weights[lidx][i * layers[lidx].input_size + j] / actual_batch_size as f64;
                        }
                        batch_delta_biases[lidx][i] += delta_biases[lidx][i] / actual_batch_size as f64;
                    }
                }
            }

            // Update
            for lidx in 0..layers.len()
            {
                for i in 0..layers[lidx].output_size
                {
                    for j in 0..layers[lidx].input_size
                    {
                        layers[lidx].weights[i * layers[lidx].input_size + j] -=
                            batch_delta_weights[lidx][i * layers[lidx].input_size + j] * eta;
                    }
                    layers[lidx].biases[i] -= batch_delta_biases[lidx][i] * eta;
                }
            }

        }

        // Validate
        let mut num_correct_data = 0;
        for i in 0..test_labels.len()
        {
            let output = nn_run(&layers, &test_data[i]);

            let mut max_index = 0;
            let mut max_value = output[0];
            for j in 1..test_labels[i].len()
            {
                if output[j] > max_value
                {
                    max_value = output[j];
                    max_index = j;
                }
            }

            if test_labels[i][max_index] == 1.0
            {
                num_correct_data += 1;
            }
        }

        let time_elapsed = timer.elapsed();
        println!("        Epoch done in {}.{:#03} seconds", time_elapsed.as_secs(), time_elapsed.subsec_millis());
        println!("        {} out of {} images predicted correctly!", num_correct_data, test_labels.len());
    }
}

fn read_labels<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<u8>>
{
    let mut fs_labels = std::fs::File::open(path)?;

    let mut magic_number: u32 = 0;
    let mut num_items:    u32 = 0;
    unsafe
    {
        let mut u32_bytes = std::slice::from_raw_parts_mut(&mut magic_number as *mut u32 as *mut u8, 4);
        fs_labels.read_exact(&mut u32_bytes)?;
        u32_bytes.reverse();

        assert!(magic_number == 0x801);

        let mut u32_bytes = std::slice::from_raw_parts_mut(&mut num_items as *mut u32 as *mut u8, 4);
        fs_labels.read_exact(&mut u32_bytes)?;
        u32_bytes.reverse();
    }

    let mut labels: Vec<u8> = Vec::new();
    fs_labels.read_to_end(&mut labels)?;

    assert!(labels.len() == num_items as usize);

    Ok(labels)
}

struct Images
{
    num_images: usize,
    width:      usize,
    height:     usize,
    data:       Vec<u8>,
}

fn read_images<P: AsRef<Path>>(path: P) -> std::io::Result<Images>
{
    let mut fs_images = std::fs::File::open(path)?;

    let mut magic_number: u32 = 0;
    let mut num_images:   u32 = 0;
    let mut num_rows:     u32 = 0;
    let mut num_cols:     u32 = 0;
    unsafe
    {
        let mut u32_bytes = std::slice::from_raw_parts_mut(&mut magic_number as *mut u32 as *mut u8, 4);
        fs_images.read_exact(&mut u32_bytes)?;
        u32_bytes.reverse();

        assert!(magic_number == 0x803);

        let mut u32_bytes = std::slice::from_raw_parts_mut(&mut num_images as *mut u32 as *mut u8, 4);
        fs_images.read_exact(&mut u32_bytes)?;
        u32_bytes.reverse();

        let mut u32_bytes = std::slice::from_raw_parts_mut(&mut num_rows as *mut u32 as *mut u8, 4);
        fs_images.read_exact(&mut u32_bytes)?;
        u32_bytes.reverse();

        let mut u32_bytes = std::slice::from_raw_parts_mut(&mut num_cols as *mut u32 as *mut u8, 4);
        fs_images.read_exact(&mut u32_bytes)?;
        u32_bytes.reverse();
    }

    let mut images = Images
    {
        num_images: num_images as usize,
        width:      num_cols as usize,
        height:     num_rows as usize,
        data:       Vec::new()
    };

    fs_images.read_to_end(&mut images.data)?;

    assert!(images.data.len() == (num_images * num_rows * num_cols) as usize);

    Ok(images)
}

fn main() -> std::io::Result<()>
{
    let train_labels = read_labels("data/train-labels.idx1-ubyte")?;
    let train_images = read_images("data/train-images.idx3-ubyte")?;

    let test_labels = read_labels("data/t10k-labels.idx1-ubyte")?;
    let test_images = read_images("data/t10k-images.idx3-ubyte")?;

    let image_size = train_images.width * train_images.height;

    let train_images_float = train_images.data.iter().map(
        |&x| (x as f64) / 255.0
    ).collect::<Vec<_>>();
    let train_images_float = train_images_float.chunks(image_size).map(|x| x.to_vec()).collect::<Vec<_>>();

    let test_images_float = test_images.data.iter().map(
        |&x| (x as f64) / 255.0
    ).collect::<Vec<_>>();
    let test_images_float = test_images_float.chunks(image_size).map(|x| x.to_vec()).collect::<Vec<_>>();

    let mut train_labels_one_hot = vec![vec![0.0; 10]; train_labels.len()];
    for i in 0..train_labels.len()
    {
        train_labels_one_hot[i][train_labels[i] as usize] = 1.0;
    }

    let mut test_labels_one_hot = vec![vec![0.0; 10]; test_labels.len()];
    for i in 0..test_labels.len()
    {
        test_labels_one_hot[i][test_labels[i] as usize] = 1.0;
    }

    let mut layers = vec![
        NnLinearLayer::new(784, 30),
        NnLinearLayer::new(30, 10),
    ];

    let num_epochs  = 30;
    let num_batches = 10;
    let eta = 3.0;

    println!("Training started.");
    let timer = std::time::Instant::now();
    sgd_train(
        &mut layers,
        &train_images_float, &train_labels_one_hot,
        &test_images_float,  &test_labels_one_hot,
        num_epochs, num_batches, eta
    );
    let time_elapsed = timer.elapsed();
    println!("Training {} epochs finished in {}.{:#03} seconds.", num_epochs, time_elapsed.as_secs(), time_elapsed.subsec_millis());

    Ok(())
}

