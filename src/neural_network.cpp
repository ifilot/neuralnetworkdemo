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

void NeuralNetwork::feed_forward(const std::vector<double>& a) {
    // copy input vector to activations
    cblas_dcopy(a.size(),
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

void NeuralNetwork::sgd(const std::shared_ptr<Dataset>& trainingset,
                        const std::shared_ptr<Dataset>& testset,
                        unsigned int epochs,
                        unsigned int mini_batch_size,
                        double eta) {
    // randomly build training set
    std::vector<unsigned int> batches(trainingset->size());
    for(unsigned int i=0; i<trainingset->size(); i++) {
        batches[i] = i;
    }
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(batches), std::end(batches), rng);

    for(unsigned int i=0; i<trainingset->size(); i+= mini_batch_size) {
        this->update_mini_batch(trainingset, batches, i, mini_batch_size, eta);

        if(i % 100 == 0) {
            std::cout << i << "\t" << this->evaluate(testset) << std::endl;
        }
    }
}

void NeuralNetwork::update_mini_batch(const std::shared_ptr<Dataset>& trainingset, const std::vector<unsigned int>& batches, unsigned int start, unsigned int batch_size, double eta) {
    std::vector<std::vector<double>> nabla_b_sum;
    std::vector<std::vector<double>> nabla_w_sum;

    // construct bias vectors
    for(unsigned int i=1; i<this->sizes.size(); i++) {
        nabla_b_sum.emplace_back(this->sizes[i], 0.0);
        nabla_w_sum.emplace_back(this->sizes[i-1] * this->sizes[i], 0.0);
    }

    for(unsigned int i=start; i<(start + batch_size); i++) {
        this->back_propagation(trainingset->get_input_vector(i), trainingset->get_output_vector(i));
        this->copy_nablas(nabla_b_sum, nabla_w_sum);
    }

    this->correct_network(nabla_b_sum, nabla_w_sum, batch_size, eta);
}

double NeuralNetwork::sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

double NeuralNetwork::sigmoid_prime(double z) {
    return this->sigmoid(z) * (1.0 - this->sigmoid(z));
}

void NeuralNetwork::copy_nablas(std::vector<std::vector<double> >& nabla_b_sum, std::vector<std::vector<double> >& nabla_w_sum) {
    for(unsigned int i=0; i<nabla_b_sum.size(); i++) {
        #pragma omp parallel for
        for(unsigned int j=0; j<nabla_b_sum[i].size(); j++) {
            nabla_b_sum[i][j] += nabla_b[i][j];
        }
    }

    for(unsigned int i=0; i<nabla_w_sum.size(); i++) {
        #pragma omp parallel for
        for(unsigned int j=0; j<nabla_w_sum[i].size(); j++) {
            nabla_w_sum[i][j] += nabla_w[i][j];
        }
    }
}

void NeuralNetwork::correct_network(std::vector<std::vector<double> >& nabla_b_sum, std::vector<std::vector<double> >& nabla_w_sum, unsigned int size, double eta) {
    const double factor = eta / (double)size;

    for(unsigned int i=0; i<nabla_b_sum.size(); i++) {
        #pragma omp parallel for
        for(unsigned int j=0; j<nabla_b_sum[i].size(); j++) {
            this->biases[i][j] -= factor * nabla_b_sum[i][j];
        }
    }

    for(unsigned int i=0; i<nabla_w_sum.size(); i++) {
        #pragma omp parallel for
        for(unsigned int j=0; j<nabla_w_sum[i].size(); j++) {
            this->weights[i][j] -= factor * nabla_w_sum[i][j];
        }
    }
}

unsigned int NeuralNetwork::evaluate(const std::shared_ptr<Dataset>& testset) {
    unsigned int hits = 0;

    for(unsigned int i=0; i<testset->size() / 100; i++) {
        this->feed_forward(testset->get_input_vector(i));

        auto vf = this->activations.front();
        auto vb = this->activations.back();
        // for(unsigned int j=0; j<vf.size(); j++) {
        //     std::cout << vf[j] << "\t";
        // }
        // std::cout << std::endl;

        // auto t2 = testset->get_input_vector(i);
        // for(unsigned int j=0; j<t2.size(); j++) {
        //     std::cout << t2[j] << "\t";
        // }
        // std::cout << std::endl;

        for(unsigned int j=0; j<vb.size(); j++) {
            std::cout << vb[j] << "\t";
        }
        std::cout << std::endl;

        if(testset->get_output_vector(i)[*std::max_element(this->activations.back().begin(), this->activations.back().end())] == 1) {
            hits++;
        }
    }

    return hits;
}
