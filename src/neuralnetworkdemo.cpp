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

#include <iostream>
#include <omp.h>
#include <chrono>
#include <boost/format.hpp>

int main(int argc, char* argv[]) {

    omp_set_num_threads(1);

    auto start = std::chrono::system_clock::now();

    MNISTLoader ml;
    ml.load_testset("../data/t10k-images-idx3-ubyte.gz", "../data/t10k-labels-idx1-ubyte.gz");
    ml.load_trainingset("../data/train-images-idx3-ubyte.gz", "../data/train-labels-idx1-ubyte.gz");

    auto trainingset = ml.get_trainingset();
    auto testset = ml.get_testset();

    NeuralNetwork nn({784,30,10});

    nn.sgd(trainingset, testset, 30, 10, 3.0);

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << boost::format("Total elapsed time: %f ms\n") % elapsed.count();
    std::cout << "--------------------------------------------------------------" << std::endl;

    return 0;
}
