#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstdint>
#include <cmath>

#include <random>
#include <chrono>

static auto rng = std::default_random_engine();

struct NnLinearLayer
{
    NnLinearLayer(size_t input_size_, size_t output_size_):
        input_size(input_size_), output_size(output_size_)
    {
        weights = new double[input_size * output_size];
        biases  = new double[output_size];

        std::normal_distribution<> d(0.0, 1.0);
        for(size_t i = 0; i < (input_size * output_size); ++i) weights[i] = d(rng);

        for(size_t i = 0; i < output_size; ++i) biases[i] = d(rng);
    }

    void process_input(const double* input, double* output) const
    {
        for(size_t i = 0; i < output_size; ++i)
        {
            output[i] = biases[i];
            for(size_t j = 0; j < input_size; ++j) output[i] += weights[i * input_size + j] * input[j];
        }
    }

    size_t input_size;
    size_t output_size;

    double* weights;
    double* biases;
};

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_prime(double x)
{
    double s = sigmoid(x);
    return s * (1.0 - s);
}

void nn_run(const NnLinearLayer* layers, size_t num_layers, const double* input, double** output)
{
    double *temp_input = new double[layers[0].input_size];
    memcpy(temp_input, input, layers[0].input_size * sizeof(double));

    for(size_t i = 0; i < num_layers; ++i)
    {
        double* temp_output = new double[layers[i].output_size];
        layers[i].process_input(temp_input, temp_output);
        for(size_t j = 0; j < layers[i].output_size; ++j) temp_output[j] = sigmoid(temp_output[j]);
        delete[] temp_input;
        temp_input = temp_output;
    }

    *output = temp_input;
}

void nn_backprop(
    const NnLinearLayer* layers, size_t num_layers,
    const double* input, const double* expected_output,
    double*** delta_weights, double*** delta_biases
){
    // Feedforward
    double** activation     = new double*[num_layers + 1];
    double** weighted_input = new double*[num_layers];

    activation[0] = new double[layers[0].input_size];
    memcpy(activation[0], input, layers[0].input_size * sizeof(double));

    for(size_t i = 0; i < num_layers; ++i)
    {
        weighted_input[i] = new double[layers[i].output_size];
        layers[i].process_input(activation[i], weighted_input[i]);

        activation[i + 1] = new double[layers[i].output_size];
        for(size_t j = 0; j < layers[i].output_size; ++j) activation[i + 1][j] = sigmoid(weighted_input[i][j]);
    }

    double* error = new double[layers[num_layers - 1].output_size];
    for(size_t i = 0; i < layers[num_layers - 1].output_size; ++i)
        error[i] = (activation[num_layers][i] - expected_output[i]) * sigmoid_prime(weighted_input[num_layers - 1][i]);

    double** delta_weights_ = new double*[num_layers];
    double** delta_biases_  = new double*[num_layers];

    for(size_t i = 0; i < num_layers; ++i)
    {
        size_t lidx = num_layers - 1 - i;

        const auto& layer = layers[lidx];
        delta_weights_[lidx] = new double[layer.input_size * layer.output_size];
        delta_biases_[lidx]  = new double[layer.output_size];

        for(size_t k = 0; k < layer.output_size; ++k)
        {
            for(size_t j = 0; j < layer.input_size; ++j) delta_weights_[lidx][k * layer.input_size + j] = error[k] * activation[lidx][j];
            delta_biases_[lidx][k] = error[k];
        }

        if(lidx == 0) break;

        double* new_error = new double[layer.input_size]{0};
        for(size_t j = 0; j < layer.input_size; ++j)
        {
            for(size_t k = 0; k < layer.output_size; ++k) new_error[j] += layer.weights[k * layer.input_size + j] * error[k];
            new_error[j] *= sigmoid_prime(weighted_input[lidx - 1][j]);
        }

        delete[] error;
        error = new_error;
    }

    *delta_weights = delta_weights_;
    *delta_biases  = delta_biases_;

    delete[] error;

    for(size_t i = 0; i < num_layers; ++i)
    {
        delete[] activation[i];
        delete[] weighted_input[i];
    }

    delete[] activation[num_layers];
    delete[] activation;
    delete[] weighted_input;
}

void sgd_train(
    const NnLinearLayer* layers, size_t num_layers,
    const double* train_data, const double* train_labels, size_t train_num_labels,
    const double* test_data, const double* test_labels, size_t test_num_labels,
    size_t num_epochs, size_t batch_size, double eta
){
    size_t datum_size  = layers[0].input_size;
    size_t output_size = layers[num_layers - 1].output_size;

    size_t* index_batches = new size_t[train_num_labels];
    for(size_t i = 0; i < train_num_labels; ++i) index_batches[i] = i;

    double** batch_delta_weights = new double*[num_layers];
    double** batch_delta_biases  = new double*[num_layers];

    for(size_t lidx = 0; lidx < num_layers; ++lidx)
    {
        batch_delta_weights[lidx] = new double[layers[lidx].input_size * layers[lidx].output_size];
        batch_delta_biases[lidx]  = new double[layers[lidx].output_size];
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
                memset(batch_delta_weights[lidx], 0, layers[lidx].input_size * layers[lidx].output_size * sizeof(double));
                memset(batch_delta_biases[lidx], 0, layers[lidx].output_size * sizeof(double));
            }

            for(size_t si = batch_start; si < batch_end; ++si)
            {
                size_t sample_index = index_batches[si];

                double **delta_weights, **delta_biases;
                nn_backprop(
                    layers, num_layers,
                    &train_data[datum_size * sample_index], &train_labels[output_size * sample_index],
                    &delta_weights, &delta_biases
                );

                // Accumulate
                size_t actual_batch_size = batch_end - batch_start;
                for(size_t lidx = 0; lidx < num_layers; ++lidx)
                {
                    for(size_t i = 0; i < layers[lidx].output_size; ++i)
                    {
                        for(size_t j = 0; j < layers[lidx].input_size; ++j)
                        {
                            batch_delta_weights[lidx][i * layers[lidx].input_size + j] +=
                                delta_weights[lidx][i * layers[lidx].input_size + j] / actual_batch_size;
                        }
                        batch_delta_biases[lidx][i] += delta_biases[lidx][i] / actual_batch_size;
                    }
                    delete[] delta_weights[lidx];
                    delete[] delta_biases[lidx];
                }

                delete[] delta_weights;
                delete[] delta_biases;
            }

            // Update
            for(size_t lidx = 0; lidx < num_layers; ++lidx)
            {
                for(size_t i = 0; i < layers[lidx].output_size; ++i)
                {
                    for(size_t j = 0; j < layers[lidx].input_size; ++j)
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
            double* output;
            nn_run(layers, num_layers, &test_data[datum_size * i], &output);

            size_t max_index = std::distance(output, std::max_element(output, output + output_size));
            if(test_labels[output_size * i + max_index] == 1.0) num_correct_data += 1;

            delete[] output;
        }

        double time_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
        printf("        Epoch done in %.3f seconds\n", time_elapsed);
        printf("        %llu out of %llu images predicted correctly!\n", num_correct_data, test_num_labels);
    }


    for(size_t lidx = 0; lidx < num_layers; ++lidx)
    {
        delete[] batch_delta_weights[lidx];
        delete[] batch_delta_biases[lidx];
    }
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

    double* train_images_float   = nullptr;
    double* train_labels_one_hot = nullptr;
    size_t train_num_labels = 0;
    size_t train_image_size = 0, train_num_images = 0;
    {
        uint8_t *images = nullptr, *labels = nullptr;

        read_labels("data/train-labels.idx1-ubyte", &labels, &train_num_labels);
        read_images("data/train-images.idx3-ubyte", &images, &train_image_size, &train_num_images);

        assert(train_num_labels == train_num_images);

        train_labels_one_hot = new double[train_num_labels * 10];
        train_images_float   = new double[train_num_images * train_image_size];
        for(size_t i = 0; i < train_num_images; ++i)
        {
            for(size_t j = 0; j < train_image_size; ++j)
                train_images_float[train_image_size * i + j] = images[train_image_size * i + j] / 255.0;

            for(size_t j = 0; j < 10; ++j) train_labels_one_hot[10 * i + j] = 0.0;
            train_labels_one_hot[10 * i + labels[i]] = 1.0;
        }
    }

    double* test_images_float   = nullptr;
    double* test_labels_one_hot = nullptr;
    size_t test_num_labels = 0;
    size_t test_image_size = 0, test_num_images = 0;
    {
        uint8_t *images = nullptr, *labels = nullptr;

        read_labels("data/t10k-labels.idx1-ubyte", &labels, &test_num_labels);
        read_images("data/t10k-images.idx3-ubyte", &images, &test_image_size, &test_num_images);

        assert(test_num_labels == test_num_images);

        test_labels_one_hot = new double[test_num_labels * 10];
        test_images_float   = new double[test_num_images * test_image_size];
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
    double eta = 3.0;

    printf("Training started.\n");
    auto start_time = std::chrono::high_resolution_clock::now();
    sgd_train(
        layers, 2,
        train_images_float, train_labels_one_hot, train_num_labels,
        test_images_float,  test_labels_one_hot, test_num_labels,
        num_epochs, num_batches, eta
    );
    double time_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    printf("Training %llu epochs finished in %.3f seconds.", num_epochs, time_elapsed);


    return 0;
}
