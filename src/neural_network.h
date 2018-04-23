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

#ifndef _NEURAL_NETWORK_H
#define _NEURAL_NETWORK_H

#include <vector>
#include <iostream>
#include <openblas/cblas.h>
#include <cmath>
#include <random>
#include <memory>
#include <algorithm>

#include "dataset.h"

class NeuralNetwork {
private:
    unsigned int num_layers;
    std::vector<unsigned int> sizes;

    // biases and weights
    std::vector<std::vector<double> > biases;
    std::vector<std::vector<double> > weights;

    // derivatives
    std::vector<std::vector<double> > nabla_b;
    std::vector<std::vector<double> > nabla_w;

    std::vector<std::vector<double> > activations;
    std::vector<std::vector<double> > z;

public:
    NeuralNetwork(const std::vector<unsigned int>& _sizes);

    void feed_forward(const std::vector<double>& a);

    void back_propagation(const std::vector<double>& x, const std::vector<double>& y);

    void sgd(const std::shared_ptr<Dataset>& dataset, const std::shared_ptr<Dataset>& testset, unsigned int epochs, unsigned int mini_batch_size, double eta);

    inline const std::vector<double>& get_output() const {
        return this->activations.back();
    }

    inline void set_biases(const std::vector<std::vector<double> >& _biases) {
        this->biases = _biases;
    }

    inline void set_weights(const std::vector<std::vector<double> >& _weights) {
        this->weights = _weights;
    }

    inline const auto& get_nabla_w() const {
        return this->nabla_w;
    }

    inline const auto& get_nabla_b() const {
        return this->nabla_b;
    }

    inline const auto& get_z() const {
        return this->z;
    }

private:
    void build_network();

    double sigmoid(double z);

    double sigmoid_prime(double z);

    void update_mini_batch(const std::shared_ptr<Dataset>& dataset, const std::vector<unsigned int>& batches, unsigned int start, unsigned int batch_size, double eta);

    void copy_nablas(std::vector<std::vector<double> >& nabla_b_sum, std::vector<std::vector<double> >& nabla_w_sum);

    void correct_network(std::vector<std::vector<double> >& nabla_b_sum, std::vector<std::vector<double> >& nabla_w_sum, unsigned int size, double eta);

    unsigned int evaluate(const std::shared_ptr<Dataset>& testset);
};

#endif // _NEURAL_NETWORK_H
