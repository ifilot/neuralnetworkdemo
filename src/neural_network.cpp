/************************************************************************************
 *   CMakeLists.txt  --  This file is part of neuralnetworkdemo.                    *
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

#include "neural_network.h"

NeuralNetwork::NeuralNetwork(const std::vector<unsigned int>& _sizes) :
sizes(_sizes) {
    this->num_layers = this->sizes.size();
    this->build_network();
}

void NeuralNetwork::build_network() {
    // construct random number generator
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::default_random_engine re;

    // construct bias vectors with zeros
    for(unsigned int i=1; i<this->sizes.size(); i++) {
        this->biases.emplace_back(this->sizes[i], 0.0);
        for(unsigned int j=0; j<this->sizes[i]; j++) {
            this->biases.back()[j] = unif(re);
        }
    }

    // construct weight matrices with zeros
    for(unsigned int i=1; i<this->sizes.size(); i++) {
        this->weights.emplace_back(this->sizes[i-1] * this->sizes[i], 0.0);
        for(unsigned int j=0; j<this->weights.back().size(); j++) {
            this->weights.back()[j] = unif(re);
        }
    }

    // construct hidden values with zeros
    for(unsigned int i=1; i<this->sizes.size() - 1; i++) {
        this->hidden_values.emplace_back(this->sizes[i], 0.0);
    }
}

std::vector<double>* NeuralNetwork::feed_forward(const std::vector<double>& a) {
    for(unsigned int i=0; i<this->hidden_values.size(); i++) {
        // copy bias vector
        cblas_dcopy(this->biases[i].size(),
                    &this->biases[i][0],
                    1,
                    &this->hidden_values[i][0],
                    1
                    );

        if(i == 0) {
            // perform matrix vector operation
            cblas_dgemv(CblasRowMajor,
                        CblasNoTrans,
                        this->hidden_values[i].size(),  // number of rows of matrix
                        a.size(),                       // number of columns of matrix
                        1.0,                            // alpha value
                        &this->weights[i][0],           // element 0 of matrix,
                        a.size(),                       // leading dimension
                        &a[0],                          // element 0 of x vector
                        1,                              // increment
                        1.0,                            // beta
                        &this->hidden_values[i][0],     // element 0 of y-vector
                        1                               // increment
                        );
        } else {
            // perform matrix vector operation
            cblas_dgemv(CblasRowMajor,
                        CblasNoTrans,
                        this->hidden_values[i].size(),      // number of rows of matrix
                        this->hidden_values[i-1].size(),    // number of columns of matrix
                        1.0,                                // alpha value
                        &this->weights[i][0],               // element 0 of matrix,
                        this->hidden_values[i-1].size(),    // leading dimension
                        &this->hidden_values[i-1][0],       // element 0 of x vector
                        1,                                  // increment
                        1.0,                                // beta
                        &this->hidden_values[i][0],         // element 0 of y-vector
                        1                                   // increment
                        );
        }
    }

    // perform matrix vector operation
    auto ans = new std::vector<double>(this->biases.back());
    cblas_dgemv(CblasRowMajor,
                CblasNoTrans,
                ans->size(),                        // number of rows of matrix
                this->hidden_values.back().size(),  // number of columns of matrix
                1.0,                                // alpha value
                &this->weights.back()[0],           // element 0 of matrix,
                this->hidden_values.back().size(),  // leading dimension
                &this->hidden_values.back()[0],     // element 0 of x vector
                1,                                  // increment
                1.0,                                // beta
                &ans->at(0),                        // element 0 of y-vector
                1                                   // increment
                );

    return ans;
}

double NeuralNetwork::sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

double NeuralNetwork::sigmoid_prime(double z) {
    return this->sigmoid(z) * (1.0 - this->sigmoid(z));
}
