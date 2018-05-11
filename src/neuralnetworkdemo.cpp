/************************************************************************************
 *   This file is part of neuralnetworkdemo.                                        *
 *   https://github.com/ifilot/neuralnetworkdemo                                    *
 *                                                                                  *
 *   MIT License                                                                    *
 *                                                                                  *
 *   Copyright (c) 2018 Ivo Filot <ivo@ivofilot.nl>                                 *
 *                                                                                  *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy   *
 *   of this software and associated documentation files (the "Software"), to deal  *
 *   in the Software without restriction, including without limitation the rights   *
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      *
 *   copies of the Software, and to permit persons to whom the Software is          *
 *   furnished to do so, subject to the following conditions:                       *
 *                                                                                  *
 *   The above copyright notice and this permission notice shall be included in all *
 *   copies or substantial portions of the Software.                                *
 *                                                                                  *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       *
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    *
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         *
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  *
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  *
 *   SOFTWARE.                                                                      *
 *                                                                                  *
 ************************************************************************************/

#include "config.h"
#include "neural_network.h"
#include "mnist_loader.h"
#include "pngfuncs.h"

#include <memory>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <boost/format.hpp>
#include <tclap/CmdLine.h>

int main(int argc, char* argv[]) {

    omp_set_num_threads(1);

    try {

        TCLAP::CmdLine cmd("neuralnetworkdemo", ' ', VERSION);

        // input file
        TCLAP::ValueArg<std::string> arg_input("i","input","Input file",false,"","filename");
        cmd.add(arg_input);

        // density file
        TCLAP::ValueArg<std::string> arg_output("o","output","Output file",false,"","filename");
        cmd.add(arg_output);

        // density file
        TCLAP::ValueArg<std::string> arg_image("f","image","Image file",false,"","filename");
        cmd.add(arg_image);

        // lua file
        TCLAP::SwitchArg arg_train("t","train","whether to further train network");
        cmd.add(arg_train);

        cmd.parse(argc, argv);

        bool train = arg_train.getValue();
        const std::string input_filename = arg_input.getValue();
        const std::string output_filename = arg_output.getValue();
        const std::string image_filename = arg_image.getValue();

        if(train) {
            auto start = std::chrono::system_clock::now();

            if(output_filename.empty()) {
                throw std::runtime_error("You need to specify an output file");
            }

            MNISTLoader ml;
            ml.load_testset("../data/t10k-images-idx3-ubyte.gz", "../data/t10k-labels-idx1-ubyte.gz");
            ml.load_trainingset("../data/train-images-idx3-ubyte.gz", "../data/train-labels-idx1-ubyte.gz");

            auto trainingset = ml.get_trainingset();
            auto testset = ml.get_testset();

            std::unique_ptr<NeuralNetwork> nn;

            if(input_filename.empty()) {
                nn = std::make_unique<NeuralNetwork>(std::vector<uint32_t>({784,30,10}));
            } else {
                std::cout << "Loading network from: " << input_filename << std::endl;
                nn = std::make_unique<NeuralNetwork>(input_filename);
            }

            nn->sgd(trainingset, testset, 10, 10, 3.0);

            std::cout << "Writing to " << output_filename << std::endl;
            nn->save_network(output_filename);

            auto end = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << boost::format("Total elapsed time: %f ms\n") % elapsed.count();
            std::cout << "--------------------------------------------------------------" << std::endl;
        } else {
            /*
             * Read sample image
             */
            if(input_filename.empty()) {
                throw std::runtime_error("You need to specify an input file for the network");
            }

            // load neural network from file
            NeuralNetwork nn(input_filename);

            // grab image
            std::cout << "Reading " << image_filename << std::endl;
            std::vector<uint8_t> buffer;
            png_uint_32 width, height;
            int col, bit_depth;
            PNG::load_image_buffer_from_png(image_filename, buffer, &width, &height, &col, &bit_depth);

            if(width != 28 || height != 28) {
                throw std::runtime_error("Image needs to be 28x28 px!");
            }

            if(col != PNG_COLOR_TYPE_GRAY) {
                throw std::runtime_error("Image needs to be saved in grayscale with no alpha channel!");
            }

            // conver to input structure
            std::vector<double> in;
            for(unsigned int i=0; i<buffer.size(); i++) {
                in.push_back((double)buffer[i] / 255.0);
            }

            // perform feed forward and output result
            nn.feed_forward(in);
            auto v = nn.get_output();
            std::cout << "--------------------------------------------------------------" << std::endl;
            std::cout << "This image is classified as \"";
            std::cout << std::distance(v.begin(), std::max_element(v.begin(), v.end()));
            std::cout << "\"" << std::endl;
            std::cout << "--------------------------------------------------------------" << std::endl;
        }

    } catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() <<
                     " for arg " << e.argId() << std::endl;
        return -1;
    }

    return 0;
}
