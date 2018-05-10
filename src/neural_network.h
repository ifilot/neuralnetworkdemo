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
#include <fstream>
#include <openblas/cblas.h>
#include <cmath>
#include <random>
#include <memory>
#include <algorithm>
#include <chrono>
#include <boost/format.hpp>

#include "dataset.h"

class NeuralNetwork {
private:
    uint32_t num_layers;                                //!< number of layers
    std::vector<uint32_t> sizes;                        //!< size of the layers

    // biases and weights
    std::vector<std::vector<double> > biases;           //!< biases
    std::vector<std::vector<double> > weights;          //!< weights

    // derivatives
    std::vector<std::vector<double> > nabla_b;          //!< bias derivative
    std::vector<std::vector<double> > nabla_w;          //!< weight derivative

    std::vector<std::vector<double> > activations;      //!< activations
    std::vector<std::vector<double> > z;                //!< signals

public:
    /**
     * @brief      Constructs a neural network
     *
     * @param[in]  _sizes  vector holding layer sizes
     */
    NeuralNetwork(const std::vector<uint32_t>& _sizes);

    /**
     * @brief      Construct a neural network
     *
     * @param[in]  filename  .net file
     */
    NeuralNetwork(const std::string& filename);

    /**
     * @brief      Perform feed forward
     *
     * @param[in]  a     input vector
     */
    void feed_forward(const std::vector<double>& a);

    /**
     * @brief      Perform back propagation
     *
     * @param[in]  x     input vector
     * @param[in]  y     expected output
     */
    void back_propagation(const std::vector<double>& x, const std::vector<double>& y);

    /**
     * @brief      Perform stochastic gradient descent
     *
     * @param[in]  dataset          training dataset
     * @param[in]  testset          test dataset
     * @param[in]  epochs           number of epochs
     * @param[in]  mini_batch_size  batch size
     * @param[in]  eta              learning rate
     */
    void sgd(const std::shared_ptr<Dataset>& dataset, const std::shared_ptr<Dataset>& testset, unsigned int epochs, unsigned int mini_batch_size, double eta);

    /**
     * @brief      save network to file
     *
     * @param[in]  filename  The filename
     */
    void save_network(const std::string& filename);

    /**
     * @brief      load network from filename
     *
     * @param[in]  filename  The filename
     */
    void load_network(const std::string& filename);

    /**
     * @brief      Gets the output.
     *
     * @return     The output.
     */
    inline const std::vector<double>& get_output() const {
        return this->activations.back();
    }

    /**
     * @brief      Sets the biases.
     *
     * @param[in]  _biases  The biases
     */
    inline void set_biases(const std::vector<std::vector<double> >& _biases) {
        this->biases = _biases;
    }

    /**
     * @brief      Sets the weights.
     *
     * @param[in]  _weights  The weights
     */
    inline void set_weights(const std::vector<std::vector<double> >& _weights) {
        this->weights = _weights;
    }

    /**
     * @brief      Gets the nabla w.
     *
     * @return     The nabla w.
     */
    inline const auto& get_nabla_w() const {
        return this->nabla_w;
    }

    /**
     * @brief      Gets the nabla b.
     *
     * @return     The nabla b.
     */
    inline const auto& get_nabla_b() const {
        return this->nabla_b;
    }

    /**
     * @brief      Gets the z.
     *
     * @return     The z.
     */
    inline const auto& get_z() const {
        return this->z;
    }

    /**
     * @brief      evaluate performance of network
     *
     * @param[in]  testset  testset
     *
     * @return     number of successful recognitions
     */
    unsigned int evaluate(const std::shared_ptr<Dataset>& testset);

private:
    /**
     * @brief      construct bias and weight vectors
     */
    void construct_bias_and_weight_vectors();

    /**
     * @brief      construct activation vectors
     */
    void construct_activation_vectors();

    /**
     * @brief      sigmoid function
     *
     * @param[in]  z     input value
     *
     * @return     sigmoid value
     */
    double sigmoid(double z);

    /**
     * @brief      derivative of sigmoid function
     *
     * @param[in]  z     input value
     *
     * @return     sigmoid derivative value
     */
    double sigmoid_prime(double z);

    /**
     * @brief      update network based on mini batch
     *
     * @param[in]  trainingset  pointer to training set
     * @param[in]  batches      pointer to mini batches
     * @param[in]  start        starting index
     * @param[in]  batch_size   batch size
     * @param[in]  eta          learning rate
     */
    void update_mini_batch(const std::shared_ptr<Dataset>& dataset, const std::vector<unsigned int>& batches, unsigned int start, unsigned int batch_size, double eta);

    /**
     * @brief      copy nablas
     *
     * @param      nabla_b_sum  The nabla b sum
     * @param      nabla_w_sum  The nabla w sum
     */
    void copy_nablas(std::vector<std::vector<double> >& nabla_b_sum, std::vector<std::vector<double> >& nabla_w_sum);

    /**
     * @brief      correct network using nabla sums
     *
     * @param[in]  nabla_b_sum  nabla b sum
     * @param[in]  nabla_w_sum  nabla w sum
     * @param[in]  batch_size   batch size
     * @param[in]  eta          learning rate
     */
    void correct_network(const std::vector<std::vector<double> >& nabla_b_sum, const std::vector<std::vector<double> >& nabla_w_sum, unsigned int batch_size, double eta);
};

#endif // _NEURAL_NETWORK_H
