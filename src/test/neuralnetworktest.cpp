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
void NeuralNetworkTest::testFeedForward() {
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

    nn.feed_forward({1.0, 2.0, 3.0});
    const auto& v = nn.get_output();

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.880537223790112, v[0], tol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.952462296779165, v[1], tol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.976593542158998,  v[2], tol);
}

/**
 * @brief      list of tests
 */
void NeuralNetworkTest::testBackPropagation() {
    static const double tol = 1e-8;
    static const double tol2 = 1e-12;

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

    std::vector<double> in = {1.0, 2.0, 3.0};
    std::vector<double> out = {1.0, 2.0, 3.0};

    nn.back_propagation(in, out);
    const auto& v = nn.get_output();

    // check cost function
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.11946278, v[0] - out[0], tol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0475377,  v[1] - out[1], tol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-2.02340646, v[2] - out[2], tol);

    // check values before sigmoid
    const auto& z = nn.get_z();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.99752738, z.back()[0], tol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.99752737, z.back()[1], tol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.73105858, z.back()[2], tol);

    // check nabla_b
    const auto& nabla_b = nn.get_nabla_b();

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.16841543e-10, nabla_b[0][0], tol2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-5.25739882e-19, nabla_b[0][1], tol2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.00000000e+00, nabla_b[0][2], tol2);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-3.09952979e-05, nabla_b[1][0], tol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.16987323e-04, nabla_b[1][1], tol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-9.09374034e-03, nabla_b[1][2], tol);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.01256646, nabla_b.back()[0], tol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.04743028, nabla_b.back()[1], tol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.04625223, nabla_b.back()[2], tol);

    // check nabla_w
    const auto& nabla_w = nn.get_nabla_w();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.16841543e-10, nabla_w.front()[0], tol2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-2.33683086e-10, nabla_w.front()[1], tol2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-3.50524629e-10, nabla_w.front()[2], tol2);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.01253539, nabla_w.back()[0], tol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.01253539, nabla_w.back()[1], tol);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.00918682, nabla_w.back()[2], tol);
}
