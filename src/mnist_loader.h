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

#ifndef _MNISTLOADER_H
#define _MNISTLOADER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <byteswap.h>

#include "pngfuncs.h"

class MNISTLoader {
private:
    size_t trainingset_size;
    std::vector<char> trainingset;
    std::vector<char> traininglabels;

    size_t testset_size;
    std::vector<char> testset;
    std::vector<char> testlabels;

public:
    MNISTLoader();

    inline size_t get_trainingset_size() const {
        return this->trainingset_size;
    }

    inline size_t get_testset_size() const {
        return this->testset_size;
    }

    void load_trainingset(const std::string& datafile, const std::string& labelfile);

    void load_testset(const std::string& datafile, const std::string& labelfile);

    void write_img_to_png(unsigned int imgid, const std::string& filename);

private:
    void load_file_from_gz(const std::string& filename, std::vector<char>* rep);
};

#endif // _MNISTLOADER_H
