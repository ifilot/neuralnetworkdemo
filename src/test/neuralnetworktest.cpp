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

#include "neuralnetworktest.h"
#include "neural_network.h"

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION(NeuralNetworkTest);

/**
 * @brief      test setup */
void NeuralNetworkTest::setUp(){}

/**
 * @brief      test tear down
 */
void NeuralNetworkTest::tearDown(){}

/**
 * @brief      list of tests
 */
void NeuralNetworkTest::testNetwork() {
    static const double tol = 1e-8;

    NeuralNetwork nn({3, 3, 3, 3});

    std::vector<std::vector<double> > biases;
    biases.push_back({1.0, 2.0, 3.0});
    biases.push_back({0.0, 0.0, 0.0});
    biases.push_back({1.0, 2.0, 3.0});
    nn.set_biases(biases);

    std::vector<std::vector<double> > weights;
    weights.push_back({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    weights.push_back({1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0, 0.0, 1.0});
    weights.push_back({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    nn.set_weights(weights);

    auto v = nn.feed_forward({1.0, 2.0, 3.0});

    // {1}: 1   2   3
    // {2}: 15  34  53
    // {3}: 242 166 53
    // {4}: 243 168 56

    CPPUNIT_ASSERT_DOUBLES_EQUAL(v->at(0), 243.0, tol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(v->at(1), 168.0, tol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(v->at(2), 56.0, tol);
}
