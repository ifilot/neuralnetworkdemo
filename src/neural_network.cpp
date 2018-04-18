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

    // construct bias vectors
    for(unsigned int i=1; i<this->sizes.size(); i++) {
        this->biases.emplace_back(this->sizes[i]);
        this->nabla_b.emplace_back(this->sizes[i]);
        for(unsigned int j=0; j<this->sizes[i]; j++) {
            this->biases.back()[j] = unif(re);
        }
    }

    // construct weight matrices
    for(unsigned int i=1; i<this->sizes.size(); i++) {
        this->weights.emplace_back(this->sizes[i-1] * this->sizes[i], 0.0);
        this->nabla_w.emplace_back(this->sizes[i-1] * this->sizes[i], 0.0);
        for(unsigned int j=0; j<this->weights.back().size(); j++) {
            this->weights.back()[j] = unif(re);
        }
    }

    // construct activations vectors
    for(unsigned int i=0; i<this->sizes.size(); i++) {
        this->activations.emplace_back(this->sizes[i]);
    }

    // construct z vectors
    for(unsigned int i=1; i<this->sizes.size(); i++) {
        this->z.emplace_back(this->sizes[i]);
    }
}

void NeuralNetwork::back_propagation(const std::vector<double>& x, const std::vector<double>& y) {
    // perform feed forward operation (store results in activations)
    this->feed_forward(x);

    // perform backward propagation
    const unsigned int sz = *std::max_element(this->sizes.begin(), this->sizes.end());
    std::vector<double> delta(sz);
    std::vector<double> tdelta(sz);

    // calculate cost derivative
    #pragma omp parallel for
    for(unsigned int i=0; i<y.size(); i++) {
        delta[i] = (this->activations.back()[i] - y[i]) * this->sigmoid_prime(this->z.back()[i]);
        nabla_b.back()[i] = delta[i];
    }

    // nabla_w(n x m) = (n x 1) * (1 x m)
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                y.size(),                           // number of rows
                this->sizes.end()[-2],              // number of columns
                1,                                  // matching dimension of the two matrices
                1.0,                                // alpha
                &delta[0],                          // matrix A
                1,                                  // leading dimension a
                &this->activations.end()[-2][0],    // matrix B
                this->sizes.end()[-2],              // leading dimension b
                0.0,                                // beta
                &nabla_w.back()[0],                 // matrix C
                this->sizes.end()[-2]               // leading dimension c
                );

    for(int i=2; i<this->num_layers; i++) {
        std::vector<double> sp(this->z.end()[-i].size());

        #pragma omp parallel for
        for(unsigned int j=0; j<this->z.end()[-i].size(); j++) {
            sp[j] = this->sigmoid_prime(this->z.end()[-i][j]);
        }

        cblas_dgemv(CblasRowMajor,
                    CblasTrans,
                    this->sizes.end()[-i+1],          // number of rows of matrix
                    this->sizes.end()[-i],            // number of columns of matrix
                    1.0,                              // alpha value
                    &this->weights.end()[-i+1][0],    // element 0 of matrix,
                    this->sizes.end()[-i],            // leading dimension
                    &delta[0],                        // element 0 of x vector
                    1,                                // increment
                    0.0,                              // beta value
                    &tdelta[0],                       // element 0 of y-vector
                    1                                 // increment
                    );

        #pragma omp parallel for
        for(unsigned int j=0; j<this->z.end()[-i].size(); j++) {
            delta[j] = tdelta[j] * sp[j];
            nabla_b.end()[-i][j] = delta[j];
        }

        cblas_dgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    this->sizes.end()[-i],              // number of rows of matrix
                    this->sizes.end()[-i-1],            // number of columns of matrix
                    1,                                  // matching dimension of the two matrices
                    1.0,                                // alpha value
                    &delta[0],                          // matrix A
                    1,                                  // leading dimension a
                    &this->activations.end()[-i-1][0],  // matrix B
                    this->sizes.end()[-i-1],            // leading dimension B
                    0.0,                                // beta
                    &nabla_w.end()[-i][0],              // matrix C
                    this->sizes.end()[-i-1]             // leading dimension C
                    );
    }
}

void NeuralNetwork::feed_forward(const std::vector<double>& a) {
    // copy input vector to activations
    cblas_dcopy(this->biases.front().size(),
                &a[0],
                1,
                &this->activations.front()[0],
                1
                );

    for(unsigned int i=1; i<this->activations.size(); i++) {
        // copy bias vector
        cblas_dcopy(this->biases[i-1].size(),
                    &this->biases[i-1][0],
                    1,
                    &this->z[i-1][0],
                    1
                    );

        cblas_dgemv(CblasRowMajor,
                    CblasNoTrans,
                    this->activations[i].size(),      // number of rows of matrix
                    this->activations[i-1].size(),    // number of columns of matrix
                    1.0,                              // alpha value
                    &this->weights[i-1][0],           // element 0 of matrix,
                    this->activations[i-1].size(),    // leading dimension
                    &this->activations[i-1][0],       // element 0 of x vector
                    1,                                // increment
                    1.0,                              // beta
                    &this->z[i-1][0],                 // element 0 of y-vector
                    1                                 // increment
                    );

        #pragma omp parallel for
        for(unsigned int j=0; j<this->activations[i].size(); j++) {
            this->activations[i][j] = this->sigmoid(this->z[i-1][j]);
        }
    }
}

double NeuralNetwork::sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

double NeuralNetwork::sigmoid_prime(double z) {
    return this->sigmoid(z) * (1.0 - this->sigmoid(z));
}
