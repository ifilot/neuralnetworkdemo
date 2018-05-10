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

/**
 * @brief      Constructs a neural network
 *
 * @param[in]  _sizes  vector holding layer sizes
 */
NeuralNetwork::NeuralNetwork(const std::vector<uint32_t>& _sizes) :
sizes(_sizes) {
    this->num_layers = this->sizes.size();
    this->construct_bias_and_weight_vectors();
    this->construct_activation_vectors();
}

/**
 * @brief      Construct a neural network
 *
 * @param[in]  filename  .net file
 */
NeuralNetwork::NeuralNetwork(const std::string& filename) {
    this->load_network(filename);
    this->construct_activation_vectors();
}

/**
 * @brief      Perform feed forward
 *
 * @param[in]  a     input vector
 */
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

/**
 * @brief      Perform back propagation
 *
 * @param[in]  x     input vector
 * @param[in]  y     expected output
 */
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

/**
 * @brief      Perform stochastic gradient descent
 *
 * @param[in]  dataset          training dataset
 * @param[in]  testset          test dataset
 * @param[in]  epochs           number of epochs
 * @param[in]  mini_batch_size  batch size
 * @param[in]  eta              learning rate
 */
void NeuralNetwork::sgd(const std::shared_ptr<Dataset>& trainingset,
                        const std::shared_ptr<Dataset>& testset,
                        unsigned int epochs,
                        unsigned int mini_batch_size,
                        double eta) {

    for(unsigned int j=0; j<epochs; j++) {
        auto start = std::chrono::system_clock::now();

        std::vector<unsigned int> batches(trainingset->size());
        for(unsigned int i=0; i<trainingset->size(); i++) {
            batches[i] = i;
        }

        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(batches), std::end(batches), rng);

        for(unsigned int i=0; i<trainingset->size(); i+= mini_batch_size) {
            this->update_mini_batch(trainingset, batches, i, mini_batch_size, eta);
        }

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << (boost::format("%4i | %i / %i | %f sec.") % (j+1) % this->evaluate(testset) % testset->size() % elapsed.count()).str() << std::endl;
    }
}

/**
 * @brief      save network to file
 *
 * @param[in]  filename  The filename
 */
void NeuralNetwork::save_network(const std::string& filename) {
    // open file
    std::ofstream out(filename, std::ios::out | std::ios::binary);

    // store sizes
    out.write((char*)&this->num_layers, sizeof(uint32_t));
    for(unsigned int i=0; i<this->sizes.size(); i++) {
        out.write((char*)&this->sizes[i], sizeof(uint32_t));
    }

    // store biases
    for(unsigned int i=0; i<this->biases.size(); i++) {
        for(unsigned int j=0; j<this->biases[i].size(); j++) {
            out.write((char*)&this->biases[i][j], sizeof(double));
        }
    }

    // store weights
    for(unsigned int i=0; i<this->weights.size(); i++) {
        for(unsigned int j=0; j<this->weights[i].size(); j++) {
            out.write((char*)&this->weights[i][j], sizeof(double));
        }
    }

    out.close();
}

/**
 * @brief      load network from filename
 *
 * @param[in]  filename  The filename
 */
void NeuralNetwork::load_network(const std::string& filename) {
    // open file
    std::ifstream in(filename, std::ios::in | std::ios::binary);

    // store sizes
    in.read((char*)&this->num_layers, sizeof(uint32_t));
    this->sizes.resize(num_layers);
    for(unsigned int i=0; i<this->sizes.size(); i++) {
        in.read((char*)&this->sizes[i], sizeof(uint32_t));
    }

    // store biases
    this->biases.resize(this->num_layers - 1);
    for(unsigned int i=0; i<this->biases.size(); i++) {
        this->biases[i].resize(this->sizes[i+1]);
        for(unsigned int j=0; j<this->biases[i].size(); j++) {
            in.read((char*)&this->biases[i][j], sizeof(double));
        }
    }

    // store weights
    this->weights.resize(this->num_layers - 1);
    for(unsigned int i=0; i<this->weights.size(); i++) {
        this->weights[i].resize(this->sizes[i] * this->sizes[i+1]);
        for(unsigned int j=0; j<this->weights[i].size(); j++) {
            in.read((char*)&this->weights[i][j], sizeof(double));
        }
    }

    in.close();
}

/**
 * @brief      evaluate performance of network
 *
 * @param[in]  testset  testset
 *
 * @return     number of successful recognitions
 */
unsigned int NeuralNetwork::evaluate(const std::shared_ptr<Dataset>& testset) {
    unsigned int hits = 0;

    for(unsigned int i=0; i<testset->size(); i++) {
        this->feed_forward(testset->get_input_vector(i));

        auto vf = this->activations.front();
        auto vb = this->activations.back();

        auto max_el = std::max_element(this->activations.back().begin(), this->activations.back().end());
        unsigned int idx = std::distance(this->activations.back().begin(), max_el);

        if(testset->get_output_vector(i)[idx] == 1) {
            hits++;
        }
    }

    return hits;
}


/**
 * @brief      construct bias and weight vectors
 */
void NeuralNetwork::construct_bias_and_weight_vectors() {
    // construct random number generator
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::default_random_engine re;
    re.seed(std::chrono::system_clock::now().time_since_epoch().count());

    // construct bias vectors
    for(unsigned int i=1; i<this->sizes.size(); i++) {
        this->biases.emplace_back(this->sizes[i]);
        for(unsigned int j=0; j<this->sizes[i]; j++) {
            this->biases.back()[j] = unif(re);
        }
    }

    // construct weight matrices
    for(unsigned int i=1; i<this->sizes.size(); i++) {
        this->weights.emplace_back(this->sizes[i-1] * this->sizes[i], 0.0);
        for(unsigned int j=0; j<this->weights.back().size(); j++) {
            this->weights.back()[j] = unif(re);
        }
    }
}

/**
 * @brief      construct activation vectors
 */
void NeuralNetwork::construct_activation_vectors() {
    // construct activations vectors
    for(unsigned int i=0; i<this->sizes.size(); i++) {
        this->activations.emplace_back(this->sizes[i]);
    }

    // construct z vectors
    for(unsigned int i=1; i<this->sizes.size(); i++) {
        this->z.emplace_back(this->sizes[i]);
    }

    // construct bias vectors
    for(unsigned int i=1; i<this->sizes.size(); i++) {
        this->nabla_b.emplace_back(this->sizes[i]);
    }

    // construct weight matrices
    for(unsigned int i=1; i<this->sizes.size(); i++) {
        this->nabla_w.emplace_back(this->sizes[i-1] * this->sizes[i], 0.0);
    }
}

/**
 * @brief      sigmoid function
 *
 * @param[in]  z     input value
 *
 * @return     sigmoid value
 */
double NeuralNetwork::sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

/**
 * @brief      derivative of sigmoid function
 *
 * @param[in]  z     input value
 *
 * @return     sigmoid derivative value
 */
double NeuralNetwork::sigmoid_prime(double z) {
    return this->sigmoid(z) * (1.0 - this->sigmoid(z));
}

/**
 * @brief      update network based on mini batch
 *
 * @param[in]  trainingset  pointer to training set
 * @param[in]  batches      pointer to mini batches
 * @param[in]  start        starting index
 * @param[in]  batch_size   batch size
 * @param[in]  eta          learning rate
 */
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

/**
 * @brief      copy nablas
 *
 * @param      nabla_b_sum  The nabla b sum
 * @param      nabla_w_sum  The nabla w sum
 */
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

/**
 * @brief      correct network using nabla sums
 *
 * @param[in]  nabla_b_sum  nabla b sum
 * @param[in]  nabla_w_sum  nabla w sum
 * @param[in]  batch_size   batch size
 * @param[in]  eta          learning rate
 */
void NeuralNetwork::correct_network(const std::vector<std::vector<double> >& nabla_b_sum, const std::vector<std::vector<double> >& nabla_w_sum, unsigned int batch_size, double eta) {
    const double factor = eta / (double)batch_size;

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
