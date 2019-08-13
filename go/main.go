package main

import (
    "fmt"
    "os"
    "math"
    "math/rand"
    "encoding/binary"
    "time"
    "runtime/pprof"
    "flag"
)

func sigmoid(x float32) float32 {
    return 1.0 / (1.0 + float32(math.Exp(-float64(x))))
}

func sigmoid_prime(x float32) float32 {
    s := sigmoid(x)
    return s * (1.0 - s)
}

type NnLinearLayer struct {
    input_size  uint32
    output_size uint32

    weights []float32
    biases  []float32
}

func NewNnLinearLayer(input_size uint32, output_size uint32) NnLinearLayer {
    layer := NnLinearLayer {
        input_size:  input_size,
        output_size: output_size,
        weights: make([]float32, input_size * output_size),
        biases:  make([]float32, output_size),
    }

    for i := uint32(0); i < output_size; i++ {
        layer.biases[i] = float32(rand.NormFloat64())
        for j := uint32(0); j < input_size; j++ {
            layer.weights[input_size * i + j] = float32(rand.NormFloat64())
        }
    }

    return layer
}

func (self NnLinearLayer) process(input []float32) []float32 {
    output := make([]float32, len(self.biases))
    copy(output, self.biases)

    for i := uint32(0); i < self.output_size; i++ {
        for j := uint32(0); j < self.input_size; j++ {
            output[i] += self.weights[i * self.input_size + j] * input[j]
        }
    }
    return output
}

func nn_run(layers []NnLinearLayer, input []float32) []float32 {
    output := make([]float32, len(input))
    copy(output, input)

    for i := range layers {
        output = layers[i].process(output)
        for j := range output {
            output[j] = sigmoid(output[j])
        }
    }
    return output
}

func nn_backprop(
    layers []NnLinearLayer,
    input []float32, expected_output []float32,
) ([][]float32, [][]float32) {

    num_layers := len(layers)
    // Feedforward
    activation     := make([][]float32, num_layers + 1)
    weighted_input := make([][]float32, num_layers)

    activation[0] = make([]float32, layers[0].input_size)
    copy(activation[0], input)

    for i := range layers {
        weighted_input[i] = layers[i].process(activation[i])

        activation[i + 1] = make([]float32, layers[i].output_size)
        for j := uint32(0); j < layers[i].output_size; j++ {
            activation[i + 1][j] = sigmoid(weighted_input[i][j])
        }
    }

    activation_error := make([]float32, layers[num_layers - 1].output_size)
    for i := uint32(0); i < layers[num_layers - 1].output_size; i++ {
        activation_error[i] = (activation[num_layers][i] - expected_output[i]) * sigmoid_prime(weighted_input[num_layers - 1][i])
    }

    delta_weights := make([][]float32, num_layers);
    delta_biases  := make([][]float32, num_layers);

    for i := range layers {

        lidx := num_layers - 1 - i

        layer := &layers[lidx];
        delta_weights[lidx] = make([]float32, layer.input_size * layer.output_size)
        delta_biases[lidx]  = make([]float32, layer.output_size)

        for k := uint32(0); k < layer.output_size; k++ {
            for j := uint32(0); j < layer.input_size; j++ {
                delta_weights[lidx][k * layer.input_size + j] = activation_error[k] * activation[lidx][j]
            }
            delta_biases[lidx][k] = activation_error[k]
        }

        if lidx == 0 { break }

        new_activation_error := make([]float32, layer.input_size)
        for j := uint32(0); j < layer.input_size; j++ {
            for k := uint32(0); k < layer.output_size; k++ {
                new_activation_error[j] += layer.weights[k * layer.input_size + j] * activation_error[k]
            }
            new_activation_error[j] *= sigmoid_prime(weighted_input[lidx - 1][j])
        }

        activation_error = new_activation_error;
    }

    return delta_weights, delta_biases
}

func sgd_train(
    layers []NnLinearLayer,
    train_data [][]float32, train_labels [][]float32,
    test_data  [][]float32, test_labels  [][]float32,
    num_epochs uint32, batch_size uint32, eta float32,
){
    num_labels  := uint32(len(train_labels))
    num_layers  := uint32(len(layers))

    index_batches := make([]uint32, num_labels)

    batch_delta_weights := make([][]float32, num_layers)
    batch_delta_biases  := make([][]float32, num_layers)

    for lidx := range layers {
        batch_delta_weights[lidx] = make([]float32, layers[lidx].input_size * layers[lidx].output_size)
        batch_delta_biases[lidx]  = make([]float32, layers[lidx].output_size)
    }

    for ei := uint32(0); ei < num_epochs; ei++ {
        fmt.Printf("    Training epoch %v\n", ei)

        start_time := time.Now()

        for i := range train_labels { index_batches[i] = uint32(i) }

        rand.Shuffle(len(index_batches), func(i, j int) {
            index_batches[i], index_batches[j] = index_batches[j], index_batches[i]
        })

        for batch_start := uint32(0); batch_start < num_labels; batch_start += batch_size {
            batch_end := batch_start + batch_size
            if batch_end > num_labels { batch_end = num_labels }

            // Clear
            for lidx := range batch_delta_weights  {
                for i := uint32(0); i < layers[lidx].output_size; i++ {
                    batch_delta_biases[lidx][i]  = 0.0
                    for j := uint32(0); j < layers[lidx].input_size; j++ {
                        batch_delta_weights[lidx][i * layers[lidx].input_size + j] = 0.0
                    }
                }
            }

            for si := batch_start; si < batch_end; si++ {
                sample_index := index_batches[si]

                delta_weights, delta_biases := nn_backprop(
                    layers,
                    train_data[sample_index],
                    train_labels[sample_index],
                )

                // Accumulate
                actual_batch_size := batch_end - batch_start
                for lidx := range layers {
                    for i := uint32(0); i < layers[lidx].output_size; i++ {
                        for j := uint32(0); j < layers[lidx].input_size; j++ {
                            batch_delta_weights[lidx][i * layers[lidx].input_size + j] +=
                                delta_weights[lidx][i * layers[lidx].input_size + j] / float32(actual_batch_size);
                        }
                        batch_delta_biases[lidx][i] += delta_biases[lidx][i] / float32(actual_batch_size);
                    }
                }
            }

            // Update
            for lidx := range layers {
                for i := uint32(0); i < layers[lidx].output_size; i++ {
                    for j := uint32(0); j < layers[lidx].input_size; j++ {
                        layers[lidx].weights[i * layers[lidx].input_size + j] -=
                            batch_delta_weights[lidx][i * layers[lidx].input_size + j] * eta;
                    }
                    layers[lidx].biases[i] -= batch_delta_biases[lidx][i] * eta;
                }
            }

        }

        // Validate
        num_correct_data := uint32(0);
        for i := range test_labels {
            output := nn_run(layers, test_data[i])

            max_index := 0;
            max_value := output[0];
            for j := range test_labels[i] {
                if output[j] > max_value {
                    max_value = output[j]
                    max_index = j
                }
            }

            if test_labels[i][max_index] == 1.0 {
                num_correct_data += 1
            }
        }

        time_elapsed := time.Now().Sub(start_time);
        fmt.Printf("        Epoch done in %v seconds\n", time_elapsed);
        fmt.Printf("        %v out of %v images predicted correctly!\n", num_correct_data, len(test_labels));
    }
}

func read_labels(filepath string) ([]uint8, error) {
    f, err := os.Open(filepath)
    if err != nil { return nil, err }
    defer f.Close()

    var magic_number uint32
    binary.Read(f, binary.BigEndian, &magic_number)
    if magic_number != 0x801 { return nil, fmt.Errorf("Magic number not correct") }

    var num_items uint32
    binary.Read(f, binary.BigEndian, &num_items)

    label_data := make([]uint8, num_items)
    count, err := f.Read(label_data)

    if err != nil { return nil, err }
    if uint32(count) != num_items { return nil, fmt.Errorf("Could not read all items") }

    return label_data, nil
}

func read_images(filepath string) ([]uint8, uint32, error) {
    f, err := os.Open(filepath)
    if err != nil { return nil, 0, err }
    defer f.Close()

    var magic_number uint32
    binary.Read(f, binary.BigEndian, &magic_number)
    if magic_number != 0x803 { return nil, 0, fmt.Errorf("Magic number not correct") }

    var num_items uint32
    binary.Read(f, binary.BigEndian, &num_items)

    var num_rows uint32
    binary.Read(f, binary.BigEndian, &num_rows)

    var num_cols uint32
    binary.Read(f, binary.BigEndian, &num_cols)

    image_size := num_rows * num_cols
    num_bytes := num_items * image_size
    image_data := make([]uint8, num_bytes)
    count, err := f.Read(image_data)

    if err != nil { return nil, 0, err }
    if uint32(count) != num_bytes { return nil, 0, fmt.Errorf("Could not read all items") }

    return image_data, image_size, nil
}

func main() {
    cpuprofile := flag.String("cpuprofile", "", "write cpu profile to `file`")
    flag.Parse()

    if *cpuprofile != "" {
        f, err := os.Create(*cpuprofile)
        if err != nil {
            panic(err)
        }
        defer f.Close()
        if err := pprof.StartCPUProfile(f); err != nil {
            panic(err)
        }
        defer pprof.StopCPUProfile()
    }

    var train_labels_one_hot [][]float32
    var train_images_float   [][]float32
    {
        label_data, err := read_labels("../cpp/data/train-labels.idx1-ubyte")
        if err != nil { panic(err) }

        image_data, image_size, err := read_images("../cpp/data/train-images.idx3-ubyte")
        if err != nil { panic(err) }

        num_images := uint32(len(image_data)) / image_size

        train_labels_one_hot = make([][]float32, num_images)
        train_images_float   = make([][]float32, num_images)

        for i := uint32(0); i < num_images; i++ {
            train_images_float[i]   = make([]float32, image_size)
            train_labels_one_hot[i] = make([]float32, 10)
            train_labels_one_hot[i][label_data[i]] = 1.0

            for j := uint32(0); j < image_size; j++ {
                train_images_float[i][j] = float32(image_data[image_size * i + j]) / 255.0
            }

        }
    }

    var test_labels_one_hot [][]float32
    var test_images_float   [][]float32
    {
        label_data, err := read_labels("../cpp/data/t10k-labels.idx1-ubyte")
        if err != nil { panic(err) }

        image_data, image_size, err := read_images("../cpp/data/t10k-images.idx3-ubyte")
        if err != nil { panic(err) }

        num_images := uint32(len(image_data)) / image_size

        test_labels_one_hot = make([][]float32, num_images)
        test_images_float   = make([][]float32, num_images)

        for i := uint32(0); i < num_images; i++ {
            test_images_float[i]   = make([]float32, image_size)
            test_labels_one_hot[i] = make([]float32, 10)
            test_labels_one_hot[i][label_data[i]] = 1.0

            for j := uint32(0); j < image_size; j++ {
                test_images_float[i][j] = float32(image_data[image_size * i + j]) / 255.0
            }
        }
    }


    layers := []NnLinearLayer {
        NewNnLinearLayer(784, 30),
        NewNnLinearLayer(30, 10),
    }

    num_epochs  := uint32(2)
    num_batches := uint32(10)
    eta := float32(3.0)

    fmt.Printf("Training started.\n")

    start_time := time.Now()
    sgd_train(
        layers,
        train_images_float, train_labels_one_hot,
        test_images_float,  test_labels_one_hot,
        num_epochs, num_batches, eta,
    )
    time_elapsed := time.Now().Sub(start_time)

    fmt.Printf("Training %v epochs finished in %v seconds.\n", num_epochs, time_elapsed)

}

