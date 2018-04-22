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

#include "dataset.h"

Dataset::Dataset(unsigned int _dataset_size, unsigned int _nr_input_nodes, unsigned int _nr_output_nodes) :
dataset_size(_dataset_size),
nr_input_nodes(_nr_input_nodes),
nr_output_nodes(_nr_output_nodes)
{
    this->x.resize(dataset_size, std::vector<double>(this->nr_input_nodes));
    this->y.resize(dataset_size, std::vector<double>(this->nr_output_nodes));
}

void Dataset::set_input_vector(unsigned int i, const std::vector<double>& vals) {
    cblas_dcopy(vals.size(), &vals[0], 1, &this->x[i][0], 1);
}

void Dataset::set_output_vector(unsigned int i, const std::vector<double>& vals) {
    cblas_dcopy(vals.size(), &vals[0], 1, &this->y[i][0], 1);
}
