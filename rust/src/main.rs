use std::convert::AsRef;
use std::path::Path;
use std::io::Read;

use rand::Rng;
use rand::prelude::thread_rng;
use rand::distributions::StandardNormal;
use rand::seq::SliceRandom;

use std::arch::x86_64::_mm256_setzero_ps;
use std::arch::x86_64::_mm256_mul_ps;
use std::arch::x86_64::_mm256_loadu_ps;
use std::arch::x86_64::_mm256_fmadd_ps;
use std::arch::x86_64::_mm256_fmsub_ps;
use std::arch::x86_64::_mm256_extractf128_ps;
use std::arch::x86_64::_mm256_castps256_ps128;
use std::arch::x86_64::_mm_add_ps;
use std::arch::x86_64::_mm_add_ss;
use std::arch::x86_64::_mm_movehl_ps;
use std::arch::x86_64::_mm_shuffle_ps;
use std::arch::x86_64::_mm_cvtss_f32;
use std::arch::x86_64::_mm256_set1_ps;
use std::arch::x86_64::_mm256_storeu_ps;

struct NnLinearLayer
{
    input_size:  usize,
    output_size: usize,

    weights: Vec<f32>,
    biases:  Vec<f32>,
}

impl NnLinearLayer
{
    fn new(input_size: usize, output_size: usize) -> NnLinearLayer
    {
        let mut rng = thread_rng();
        let weights = rng.sample_iter(&StandardNormal)
                        .take(input_size * output_size).map(|x| x as f32).collect();
        let biases  = rng.sample_iter(&StandardNormal)
                        .take(output_size).map(|x| x as f32).collect();

        NnLinearLayer{
            input_size: input_size,
            output_size: output_size,
            weights: weights, biases: biases
        }
    }

    fn process_input(&self, input: &[f32]) -> Vec<f32>
    {
        assert!(input.len() == self.input_size);

        let mut output = Vec::with_capacity(self.output_size);
        let last_index = 8 * (self.input_size / 8);

        unsafe
        {
            output.set_len(self.output_size);

            for i in 0..self.output_size
            {
                let mut acc = _mm256_setzero_ps();
                for j in (0..last_index).step_by(8)
                {
                    let sse_w  = _mm256_loadu_ps(self.weights.as_ptr().offset((i * self.input_size + j) as isize));
                    let sse_in = _mm256_loadu_ps(input.as_ptr().offset(j as isize));
                    acc = _mm256_fmadd_ps(sse_w, sse_in, acc);
                }

                let temp_sum = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
                let temp_sum = _mm_add_ps(temp_sum, _mm_movehl_ps(temp_sum, temp_sum));
                let temp_sum = _mm_add_ss(temp_sum, _mm_shuffle_ps(temp_sum, temp_sum, 0x55));
                let sum = _mm_cvtss_f32(temp_sum);

                output[i] = self.biases[i] + sum;

                for j in last_index..self.input_size
                {
                    output[i] += self.weights[i * self.input_size + j] * input[j];
                }
            }
        }

        output
    }
}

fn sigmoid(x: f32) -> f32
{
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_prime(x: f32) -> f32
{
    let s = sigmoid(x);
    s * (1.0 - s)
}

fn nn_run(layers: &[NnLinearLayer], input: &[f32]) -> Vec<f32>
{
    let mut output = input.to_vec();
    for i in 0..layers.len() { output = layers[i].process_input(&output).iter().map(|&x| sigmoid(x)).collect() }
    output
}

fn sgd_train(
    layers: &mut [NnLinearLayer],
    train_data: &Vec<Vec<f32>>, train_labels: &Vec<Vec<f32>>,
    test_data:  &Vec<Vec<f32>>, test_labels:  &Vec<Vec<f32>>,
    num_epochs: usize, batch_size: usize, eta: f32
){
    let num_layers = layers.len();

    let mut index_batches = (1..train_labels.len()).collect::<Vec<_>>();

    let mut batch_delta_weights: Vec<Vec<f32>> = vec![Vec::new(); num_layers];
    let mut batch_delta_biases: Vec<Vec<f32>>  = vec![Vec::new(); num_layers];

    let mut delta_weights = vec![Vec::new(); num_layers];
    let mut delta_biases  = vec![Vec::new(); num_layers];

    let mut activation     = vec![Vec::new(); num_layers + 1];
    let mut weighted_input = vec![Vec::new(); num_layers];

    for lidx in 0..num_layers
    {
        batch_delta_weights[lidx].resize_with(layers[lidx].input_size * layers[lidx].output_size, || 0.0);
        batch_delta_biases[lidx].resize_with(layers[lidx].output_size, || 0.0);

        delta_weights[lidx].resize_with(layers[lidx].input_size * layers[lidx].output_size, || 0.0);
        delta_biases[lidx].resize_with(layers[lidx].output_size, || 0.0);
    }

    for ei in 0..num_epochs
    {
        println!("    Training epoch {}", ei);
        let timer = std::time::Instant::now();
        index_batches.shuffle(&mut thread_rng());

        for batch_indices in index_batches.chunks(batch_size)
        {

            for lidx in 0..num_layers
            {
                for v in &mut batch_delta_weights[lidx] { *v = 0.0 }
                for v in &mut batch_delta_biases[lidx]  { *v = 0.0 }
            }

            for &sample_index in batch_indices
            {
                let actual_batch_size = batch_indices.len();
                // Backprop
                {
                    // Feedforward

                    activation[0].clone_from(&train_data[sample_index]);

                    for lidx in 0..num_layers
                    {
                        weighted_input[lidx] = layers[lidx].process_input(&activation[lidx]);
                        activation[lidx + 1] = weighted_input[lidx].iter().map(|&x| sigmoid(x)).collect::<Vec<f32>>();
                    }

                    let mut error = (0..activation[num_layers].len()).map(
                        |i| (activation[num_layers][i] - train_labels[sample_index][i]) *
                            sigmoid_prime(weighted_input[num_layers - 1][i])
                    ).collect::<Vec<f32>>();

                    for lidx in (0..num_layers).rev()
                    {
                        let layer = &layers[lidx];

                        let last_index = 8 * (layer.input_size / 8);
                        for k in 0..layer.output_size
                        {
                            unsafe
                            {
                                let error_x8 = _mm256_set1_ps(error[k]);
                                for j in (0..last_index).step_by(8)
                                {
                                    let activation_x8 = _mm256_loadu_ps(activation[lidx].as_ptr().offset(j as isize));
                                    _mm256_storeu_ps(
                                        delta_weights[lidx].as_mut_ptr().offset((k * layer.input_size + j) as isize),
                                        _mm256_mul_ps(error_x8, activation_x8)
                                    );
                                }
                            }

                            for j in last_index..layer.input_size
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
                            for k in 0..layer.output_size { new_error[j] += layer.weights[k * layer.input_size + j] * error[k] }
                            new_error[j] *= sigmoid_prime(weighted_input[lidx - 1][j]);
                        }

                        error = new_error;
                    }
                }

                // Accumulate
                unsafe
                {
                    let factor = 1.0 / actual_batch_size as f32;
                    let factor_x8 = _mm256_set1_ps(factor);

                    for lidx in 0..num_layers
                    {
                        let last_index = 8 * (layers[lidx].input_size / 8);
                        for i in 0..layers[lidx].output_size
                        {
                            for j in (0..last_index).step_by(8)
                            {
                                let sse_bw = _mm256_loadu_ps(
                                    batch_delta_weights[lidx].as_ptr().offset((i * layers[lidx].input_size + j) as isize)
                                );
                                let sse_w  = _mm256_loadu_ps(
                                    delta_weights[lidx].as_ptr().offset((i * layers[lidx].input_size + j) as isize)
                                );
                                let result = _mm256_fmadd_ps(factor_x8, sse_w, sse_bw);

                                _mm256_storeu_ps(
                                    batch_delta_weights[lidx].as_mut_ptr().offset((i * layers[lidx].input_size + j) as isize),
                                    result
                                );
                            }
                            for j in last_index..layers[lidx].input_size
                            {
                                batch_delta_weights[lidx][i * layers[lidx].input_size + j] +=
                                    delta_weights[lidx][i * layers[lidx].input_size + j] * factor;
                            }
                            batch_delta_biases[lidx][i] += delta_biases[lidx][i] * factor;
                        }
                    }
                }
            }

            // Update
            unsafe
            {
                let eta_x8 = _mm256_set1_ps(eta);
                for lidx in 0..num_layers
                {
                    let last_index = 8 * (layers[lidx].input_size / 8);
                    for i in 0..layers[lidx].output_size
                    {
                        for j in (0..last_index).step_by(8)
                        {
                            let sse_bw = _mm256_loadu_ps(
                                batch_delta_weights[lidx].as_ptr().offset((i * layers[lidx].input_size + j) as isize)
                            );
                            let sse_w  = _mm256_loadu_ps(
                                layers[lidx].weights.as_ptr().offset((i * layers[lidx].input_size + j) as isize)
                            );
                            let result = _mm256_fmsub_ps(eta_x8, sse_bw, sse_w);
                            _mm256_storeu_ps(
                                layers[lidx].weights.as_mut_ptr().offset((i * layers[lidx].input_size + j) as isize),
                                result
                            );
                        }
                        for j in last_index..layers[lidx].input_size
                        {
                            layers[lidx].weights[i * layers[lidx].input_size + j] -=
                                batch_delta_weights[lidx][i * layers[lidx].input_size + j] * eta;
                        }
                        layers[lidx].biases[i] -= batch_delta_biases[lidx][i] * eta;
                    }
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
        |&x| (x as f32) / 255.0
    ).collect::<Vec<_>>();
    let train_images_float = train_images_float.chunks(train_images.image_size).map(|x| x.to_vec()).collect::<Vec<_>>();

    let test_images_float = test_images.data.iter().map(
        |&x| (x as f32) / 255.0
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

