use std::path::Path;
use std::convert::AsRef;
use std::io::Read;

use rand::Rng;
use rand::prelude::thread_rng;
use rand::distributions::StandardNormal;
use rand::seq::SliceRandom;

use ndarray::prelude::*;

struct NnLinearLayer
{
    weights: Array2<f64>,
    biases:  Array1<f64>,
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
            weights: Array::from_shape_vec((output_size, input_size), weights).unwrap(),
            biases: Array::from_vec(biases)
        }
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

fn nn_run(layers: &[NnLinearLayer], input: &ArrayView1<f64>) -> Array1<f64>
{
    let mut output = input.to_owned();
    for i in 0..layers.len() { output = (layers[i].weights.dot(&output.view()) + &layers[i].biases).mapv(sigmoid) }
    output
}

fn nn_backprop(layers: &[NnLinearLayer], input: &ArrayView1<f64>, expected_output: &ArrayView1<f64>) -> (Vec<Array2<f64>>, Vec<Array1<f64>>)
{
    let num_layers = layers.len();

    // Feedforward
    let mut activation:     Vec<Array1<f64>> = Vec::new();
    let mut weighted_input: Vec<Array1<f64>> = Vec::new();

    activation.push(input.to_owned());

    for i in 0..num_layers
    {
        let wi = layers[i].weights.dot(&activation[i].view()) + &layers[i].biases;
        let layer_activation = wi.mapv(sigmoid);
        weighted_input.push(wi);
        activation.push(layer_activation);
    }

    let mut error = (&activation[num_layers] - expected_output) * weighted_input[num_layers - 1].mapv(sigmoid_prime);

    let mut delta_weights: Vec<Array2<f64>> = Vec::new();
    let mut delta_biases:  Vec<Array1<f64>> = Vec::new();

    for lidx in (0..num_layers).rev()
    {
        let layer = &layers[lidx];
        delta_weights.push((error.view().insert_axis(ndarray::Axis(1))).dot(&activation[lidx].view().insert_axis(ndarray::Axis(0))));
        delta_biases.push(error.clone());

        if lidx == 0 { break }

        error = layer.weights.t().dot(&error) * weighted_input[lidx - 1].mapv(sigmoid_prime);
    }

    delta_weights.reverse();
    delta_biases.reverse();

    (delta_weights, delta_biases)
}

fn sgd_train(
    layers: &mut [NnLinearLayer],
    train_data: &Vec<Vec<f64>>, train_labels: &Vec<Vec<f64>>,
    test_data:  &Vec<Vec<f64>>, test_labels:  &Vec<Vec<f64>>,
    num_epochs: usize, batch_size: usize, eta: f64
){
    let mut index_batches = (1..train_labels.len()).collect::<Vec<_>>();

    let mut batch_delta_weights: Vec<Array2<f64>> = Vec::new();
    let mut batch_delta_biases:  Vec<Array1<f64>> = Vec::new();

    for lidx in 0..layers.len()
    {
        unsafe
        {
            batch_delta_weights.push(Array::uninitialized(layers[lidx].weights.raw_dim()));
            batch_delta_biases.push(Array::uninitialized(layers[lidx].biases.raw_dim()));
        }
    }

    for ei in 0..num_epochs
    {
        println!("    Training epoch {}", ei);
        let timer = std::time::Instant::now();
        index_batches.shuffle(&mut thread_rng());

        for batch_indices in index_batches.chunks(batch_size)
        {

            for lidx in 0..layers.len()
            {
                batch_delta_weights[lidx].fill(0.0);
                batch_delta_biases[lidx].fill(0.0);
            }

            for &sample_index in batch_indices
            {
                let actual_batch_size = batch_indices.len();

                let (delta_weights, delta_biases) = unsafe {
                    let train_data_view   = ArrayView::from_shape_ptr(train_data[sample_index].len(), train_data[sample_index].as_ptr());
                    let train_labels_view = ArrayView::from_shape_ptr(train_labels[sample_index].len(), train_labels[sample_index].as_ptr());

                    nn_backprop(layers, &train_data_view, &train_labels_view)
                };

                // Accumulate
                for lidx in 0..layers.len()
                {
                    batch_delta_weights[lidx] += &(&delta_weights[lidx] / (actual_batch_size as f64));
                    batch_delta_biases[lidx]  += &(&delta_biases[lidx] / (actual_batch_size as f64));
                }
            }

            // Update
            for lidx in 0..layers.len()
            {
                layers[lidx].weights -= &(&batch_delta_weights[lidx] * eta);
                layers[lidx].biases  -= &(&batch_delta_biases[lidx] * eta);
            }

        }

        // Validate
        let mut num_correct_data = 0;
        for i in 0..test_labels.len()
        {
            let output = unsafe {
                let test_data_view = ArrayView::from_shape_ptr(test_data[i].len(), test_data[i].as_ptr());
                nn_run(&layers, &test_data_view)
            };

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

            if test_labels[i][max_index] == 1.0 { num_correct_data += 1 }
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
    image_size: usize,
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
        image_size: (num_cols * num_rows) as usize,
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

    let train_images_float = train_images.data.iter().map(
        |&x| (x as f64) / 255.0
    ).collect::<Vec<_>>();
    let train_images_float = train_images_float.chunks(train_images.image_size).map(|x| x.to_vec()).collect::<Vec<_>>();

    let test_images_float = test_images.data.iter().map(
        |&x| (x as f64) / 255.0
    ).collect::<Vec<_>>();
    let test_images_float = test_images_float.chunks(test_images.image_size).map(|x| x.to_vec()).collect::<Vec<_>>();

    let mut train_labels_one_hot = vec![vec![0.0; 10]; train_labels.len()];
    for i in 0..train_labels.len() { train_labels_one_hot[i][train_labels[i] as usize] = 1.0 }

    let mut test_labels_one_hot = vec![vec![0.0; 10]; test_labels.len()];
    for i in 0..test_labels.len() { test_labels_one_hot[i][test_labels[i] as usize] = 1.0 }

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

