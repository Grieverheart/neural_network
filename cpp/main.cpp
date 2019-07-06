#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstdint>
#include <cmath>

#include <random>
#include <chrono>

#include <immintrin.h>

static auto rng = std::default_random_engine();

struct NnLinearLayer
{
    NnLinearLayer(size_t input_size_, size_t output_size_):
        input_size(input_size_), output_size(output_size_)
    {
        weights = new float[input_size * output_size];
        biases  = new float[output_size];

        std::normal_distribution<> d(0.0, 1.0);
        for(size_t i = 0; i < (input_size * output_size); ++i) weights[i] = d(rng);

        for(size_t i = 0; i < output_size; ++i) biases[i] = d(rng);
    }

    void process_input(const float* input, float* output) const
    {
        for(size_t i = 0; i < output_size; ++i)
        {
            auto acc = _mm256_setzero_ps();
            size_t last_index = 8 * (input_size / 8);
            for(size_t j = 0; j < last_index; j += 8)
            {
                auto sse_w  = _mm256_loadu_ps(weights + i * input_size + j);
                auto sse_in = _mm256_loadu_ps(input + j);
                acc = _mm256_fmadd_ps(sse_w, sse_in, acc);
            }

            auto temp_sum = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
            temp_sum = _mm_add_ps(temp_sum, _mm_movehl_ps(temp_sum, temp_sum));
            temp_sum = _mm_add_ss(temp_sum, _mm_shuffle_ps(temp_sum, temp_sum, 0x55));
            float sum = _mm_cvtss_f32(temp_sum);

            output[i] = biases[i] + sum;

            for(size_t j = last_index; j < input_size; ++j)
                output[i] += weights[i * input_size + j] * input[j];
        }
    }

    size_t input_size;
    size_t output_size;

    float* weights;
    float* biases;
};

float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

float sigmoid_prime(float x)
{
    float s = sigmoid(x);
    return s * (1.0 - s);
}

void nn_run(const NnLinearLayer* layers, size_t num_layers, const float* input, float** output)
{
    float *temp_input = new float[layers[0].input_size];
    memcpy(temp_input, input, layers[0].input_size * sizeof(float));

    for(size_t i = 0; i < num_layers; ++i)
    {
        float* temp_output = new float[layers[i].output_size];
        layers[i].process_input(temp_input, temp_output);
        for(size_t j = 0; j < layers[i].output_size; ++j) temp_output[j] = sigmoid(temp_output[j]);
        delete[] temp_input;
        temp_input = temp_output;
    }

    *output = temp_input;
}

void sgd_train(
    const NnLinearLayer* layers, size_t num_layers,
    const float* train_data, const float* train_labels, size_t train_num_labels,
    const float* test_data, const float* test_labels, size_t test_num_labels,
    size_t num_epochs, size_t batch_size, float eta
){
    size_t datum_size  = layers[0].input_size;
    size_t output_size = layers[num_layers - 1].output_size;

    size_t* index_batches = new size_t[train_num_labels];
    for(size_t i = 0; i < train_num_labels; ++i) index_batches[i] = i;

    float** batch_delta_weights = new float*[num_layers];
    float** batch_delta_biases  = new float*[num_layers];

    float** delta_weights = new float*[num_layers];
    float** delta_biases  = new float*[num_layers];

    float** activation     = new float*[num_layers + 1];
    float** weighted_input = new float*[num_layers];

    activation[0] = new float[layers[0].input_size];

    for(size_t lidx = 0; lidx < num_layers; ++lidx)
    {
        batch_delta_weights[lidx] = new float[layers[lidx].input_size * layers[lidx].output_size];
        batch_delta_biases[lidx]  = new float[layers[lidx].output_size];

        delta_weights[lidx] = new float[layers[lidx].input_size * layers[lidx].output_size];
        delta_biases[lidx]  = new float[layers[lidx].output_size];

        activation[lidx + 1] = new float[layers[lidx].input_size];
        weighted_input[lidx] = new float[layers[lidx].output_size];
    }

    for(size_t ei = 0; ei < num_epochs; ++ei)
    {
        printf("    Training epoch %llu\n", ei);

        auto start_time = std::chrono::high_resolution_clock::now();

        std::shuffle(index_batches, index_batches + train_num_labels, rng);

        for(size_t batch_start = 0; batch_start < train_num_labels; batch_start += batch_size)
        {
            size_t batch_end = (batch_start + batch_size <= train_num_labels)?
                batch_start + batch_size: train_num_labels;

            for(size_t lidx = 0; lidx < num_layers; ++lidx)
            {
                memset(batch_delta_weights[lidx], 0, layers[lidx].input_size * layers[lidx].output_size * sizeof(float));
                memset(batch_delta_biases[lidx], 0, layers[lidx].output_size * sizeof(float));
            }

            for(size_t si = batch_start; si < batch_end; ++si)
            {
                size_t sample_index = index_batches[si];

                for(size_t lidx = 0; lidx < num_layers; ++lidx)
                {
                }

                // Backprop
                {
                    const float* input = &train_data[datum_size * sample_index];
                    const float* expected_output = &train_labels[output_size * sample_index];

                    memcpy(activation[0], input, layers[0].input_size * sizeof(float));

                    // Feedforward
                    for(size_t i = 0; i < num_layers; ++i)
                    {
                        layers[i].process_input(activation[i], weighted_input[i]);
                        for(size_t j = 0; j < layers[i].output_size; ++j) activation[i + 1][j] = sigmoid(weighted_input[i][j]);
                    }

                    float* error = new float[layers[num_layers - 1].output_size];
                    for(size_t i = 0; i < layers[num_layers - 1].output_size; ++i)
                        error[i] = (activation[num_layers][i] - expected_output[i]) * sigmoid_prime(weighted_input[num_layers - 1][i]);

                    for(size_t i = 0; i < num_layers; ++i)
                    {
                        size_t lidx = num_layers - 1 - i;

                        const auto& layer = layers[lidx];

                        for(size_t k = 0; k < layer.output_size; ++k)
                        {
                            auto error_x8 = _mm256_set1_ps(error[k]);
                            for(size_t j = 0; j < 8 * (layer.input_size / 8); j += 8)
                            {
                                auto activation_x8 = _mm256_loadu_ps(activation[lidx] + j);
                                _mm256_storeu_ps(delta_weights[lidx] + k * layer.input_size + j, _mm256_mul_ps(error_x8, activation_x8));
                            }

                            for(size_t j = 8 * (layer.input_size / 8); j < layer.input_size; ++j)
                                delta_weights[lidx][k * layer.input_size + j] = error[k] * activation[lidx][j];

                            delta_biases[lidx][k] = error[k];
                        }

                        if(lidx == 0) break;

                        float* new_error = new float[layer.input_size];
                        for(size_t j = 0; j < layer.input_size; ++j)
                        {
                            new_error[j] = 0.0;

                            for(size_t k = 0; k < layer.output_size; ++k)
                                new_error[j] += layer.weights[k * layer.input_size + j] * error[k];

                            new_error[j] *= sigmoid_prime(weighted_input[lidx - 1][j]);
                        }

                        delete[] error;
                        error = new_error;
                    }

                    delete[] error;
                }

                // Accumulate
                float factor = 1.0f / (batch_end - batch_start);
                auto factor_x8 = _mm256_set1_ps(factor);
                for(size_t lidx = 0; lidx < num_layers; ++lidx)
                {
                    for(size_t i = 0; i < layers[lidx].output_size; ++i)
                    {
                        size_t last_index = 8 * (layers[lidx].input_size / 8);
                        for(size_t j = 0; j < last_index; j += 8)
                        {
                            auto sse_bw = _mm256_loadu_ps(batch_delta_weights[lidx] + i * layers[lidx].input_size + j);
                            auto sse_w = _mm256_loadu_ps(delta_weights[lidx] + i * layers[lidx].input_size + j);
                            auto result = _mm256_fmadd_ps(factor_x8, sse_w, sse_bw);
                            _mm256_storeu_ps(batch_delta_weights[lidx] + i * layers[lidx].input_size + j, result);
                        }
                        for(size_t j = last_index; j < layers[lidx].input_size; ++j)
                        {
                            batch_delta_weights[lidx][i * layers[lidx].input_size + j] +=
                                delta_weights[lidx][i * layers[lidx].input_size + j] * factor;
                        }
                        batch_delta_biases[lidx][i] += delta_biases[lidx][i] * factor;
                    }
                }
            }

            // Update
            auto eta_x8 = _mm256_set1_ps(eta);
            for(size_t lidx = 0; lidx < num_layers; ++lidx)
            {
                for(size_t i = 0; i < layers[lidx].output_size; ++i)
                {
                    size_t last_index = 8 * (layers[lidx].input_size / 8);
                    for(size_t j = 0; j < last_index; j += 8)
                    {
                        auto sse_bw = _mm256_loadu_ps(batch_delta_weights[lidx] + i * layers[lidx].input_size + j);
                        auto sse_w = _mm256_loadu_ps(layers[lidx].weights + i * layers[lidx].input_size + j);
                        auto result = _mm256_fmsub_ps(eta_x8, sse_bw, sse_w);
                        _mm256_storeu_ps(layers[lidx].weights + i * layers[lidx].input_size + j, result);
                    }
                    for(size_t j = last_index; j < layers[lidx].input_size; ++j)
                    {
                        layers[lidx].weights[i * layers[lidx].input_size + j] -=
                            batch_delta_weights[lidx][i * layers[lidx].input_size + j] * eta;
                    }
                    layers[lidx].biases[i] -= batch_delta_biases[lidx][i] * eta;
                }
            }

        }

        // Validate
        size_t num_correct_data = 0;
        for(size_t i = 0; i < test_num_labels; ++i)
        {
            float* output;
            nn_run(layers, num_layers, &test_data[datum_size * i], &output);

            size_t max_index = std::distance(output, std::max_element(output, output + output_size));
            if(test_labels[output_size * i + max_index] == 1.0) num_correct_data += 1;

            delete[] output;
        }

        float time_elapsed = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start_time).count();
        printf("        Epoch done in %.3f seconds\n", time_elapsed);
        printf("        %llu out of %llu images predicted correctly!\n", num_correct_data, test_num_labels);
    }

    for(size_t lidx = 0; lidx < num_layers; ++lidx)
    {
        delete[] batch_delta_weights[lidx];
        delete[] batch_delta_biases[lidx];

        delete[] delta_weights[lidx];
        delete[] delta_biases[lidx];

        delete[] activation[lidx];
        delete[] weighted_input[lidx];
    }

    delete[] activation[num_layers];
    delete[] activation;
    delete[] weighted_input;
    delete[] delta_weights;
    delete[] delta_biases;
    delete[] batch_delta_weights;
    delete[] batch_delta_biases;
    delete[] index_batches;
}

static uint32_t be32_to_cpu(uint32_t x) {
    uint32_t ret = (
        ((uint32_t) ((uint8_t*)&x)[3] << 0)  |
        ((uint32_t) ((uint8_t*)&x)[2] << 8)  |
        ((uint32_t) ((uint8_t*)&x)[1] << 16) |
        ((uint32_t) ((uint8_t*)&x)[0] << 24)
    );
    return *(uint32_t*)(&ret);
}

void read_labels(const char* filepath, uint8_t** label_data, size_t* num_labels)
{
    FILE* fp = fopen(filepath, "rb");

    uint32_t magic_number;
    uint32_t num_items;

    fread(&magic_number, 4, 1, fp);
    magic_number = be32_to_cpu(magic_number);
    assert(magic_number == 0x801);

    fread(&num_items, 4, 1, fp);
    num_items = be32_to_cpu(num_items);

    *label_data = new uint8_t[num_items];
    fread(*label_data, num_items, 1, fp);

    fclose(fp);

    *num_labels = num_items;
}

void read_images(const char* filepath, uint8_t** image_data, size_t* image_size, size_t* num_images)
{
    FILE* fp = fopen(filepath, "rb");

    uint32_t magic_number;
    uint32_t num_items;
    uint32_t num_rows;
    uint32_t num_cols;

    fread(&magic_number, 4, 1, fp);
    magic_number = be32_to_cpu(magic_number);
    assert(magic_number == 0x803);

    fread(&num_items, 4, 1, fp);
    num_items = be32_to_cpu(num_items);

    fread(&num_rows, 4, 1, fp);
    num_rows = be32_to_cpu(num_rows);

    fread(&num_cols, 4, 1, fp);
    num_cols = be32_to_cpu(num_cols);

    *image_data = new uint8_t[num_items * num_rows * num_cols];
    fread(*image_data, num_items * num_rows * num_cols, 1, fp);

    *num_images = num_items;
    *image_size = num_rows * num_cols;

    fclose(fp);
}

int main(int argc, char* argv[])
{
    std::random_device r;
    rng.seed(r());

    float* train_images_float   = nullptr;
    float* train_labels_one_hot = nullptr;
    size_t train_num_labels = 0;
    size_t train_image_size = 0, train_num_images = 0;
    {
        uint8_t *images = nullptr, *labels = nullptr;

        read_labels("data/train-labels.idx1-ubyte", &labels, &train_num_labels);
        read_images("data/train-images.idx3-ubyte", &images, &train_image_size, &train_num_images);

        assert(train_num_labels == train_num_images);

        train_labels_one_hot = new float[train_num_labels * 10];
        train_images_float   = new float[train_num_images * train_image_size];
        for(size_t i = 0; i < train_num_images; ++i)
        {
            for(size_t j = 0; j < train_image_size; ++j)
                train_images_float[train_image_size * i + j] = images[train_image_size * i + j] / 255.0;

            for(size_t j = 0; j < 10; ++j) train_labels_one_hot[10 * i + j] = 0.0;
            train_labels_one_hot[10 * i + labels[i]] = 1.0;
        }
    }

    float* test_images_float   = nullptr;
    float* test_labels_one_hot = nullptr;
    size_t test_num_labels = 0;
    size_t test_image_size = 0, test_num_images = 0;
    {
        uint8_t *images = nullptr, *labels = nullptr;

        read_labels("data/t10k-labels.idx1-ubyte", &labels, &test_num_labels);
        read_images("data/t10k-images.idx3-ubyte", &images, &test_image_size, &test_num_images);

        assert(test_num_labels == test_num_images);

        test_labels_one_hot = new float[test_num_labels * 10];
        test_images_float   = new float[test_num_images * test_image_size];
        for(size_t i = 0; i < test_num_images; ++i)
        {
            for(size_t j = 0; j < test_image_size; ++j)
                test_images_float[test_image_size * i + j] = images[test_image_size * i + j] / 255.0;

            for(size_t j = 0; j < 10; ++j) test_labels_one_hot[10 * i + j] = 0.0;
            test_labels_one_hot[10 * i + labels[i]] = 1.0;
        }
    }

    assert(train_image_size == test_image_size);

    auto layers = new NnLinearLayer[2]
    {
        NnLinearLayer(784, 30),
        NnLinearLayer(30, 10)
    };

    size_t num_epochs  = 30;
    size_t num_batches = 10;
    float eta = 3.0;

    printf("Training started.\n");
    auto start_time = std::chrono::high_resolution_clock::now();
    sgd_train(
        layers, 2,
        train_images_float, train_labels_one_hot, train_num_labels,
        test_images_float,  test_labels_one_hot, test_num_labels,
        num_epochs, num_batches, eta
    );
    float time_elapsed = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start_time).count();
    printf("Training %llu epochs finished in %.3f seconds.", num_epochs, time_elapsed);


    return 0;
}
