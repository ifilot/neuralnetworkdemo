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

#include "mnist_loader.h"

MNISTLoader::MNISTLoader() {

}

void MNISTLoader::load_trainingset(const std::string& datafile, const std::string& labelfile) {
    this->load_file_from_gz(datafile, &this->trainingset);
    this->load_file_from_gz(labelfile, &this->traininglabels);

    uint32_t val = 0;
    std::memcpy(&val, &this->traininglabels[0], 4);
    uint32_t checksum = __bswap_32(val);
    if(checksum != 2049) {
        throw std::runtime_error("Invalid MNIST training labels loaded");
    }

    std::memcpy(&val, &this->trainingset[0], 4);
    checksum = __bswap_32(val);
    if(checksum != 2051) {
        throw std::runtime_error("Invalid MNIST training images loaded");
    }

    std::memcpy(&val, &this->traininglabels[4], 4);
    val = __bswap_32(val);
    this->trainingset_size = val;
}

void MNISTLoader::load_testset(const std::string& datafile, const std::string& labelfile) {
    this->load_file_from_gz(datafile, &this->testset);
    this->load_file_from_gz(labelfile, &this->testlabels);

    uint32_t val = 0;
    std::memcpy(&val, &this->testlabels[0], 4);
    uint32_t checksum = __bswap_32(val);
    if(checksum != 2049) {
        throw std::runtime_error("Invalid MNIST test labels loaded");
    }

    std::memcpy(&val, &this->testset[0], 4);
    checksum = __bswap_32(val);
    if(checksum != 2051) {
        throw std::runtime_error("Invalid MNIST test images loaded");
    }

    std::memcpy(&val, &this->testlabels[4], 4);
    val = __bswap_32(val);
    this->testset_size = val;
}

void MNISTLoader::load_file_from_gz(const std::string& filename, std::vector<char>* rep) {
    // load file
    std::vector<char> raw;
    std::ifstream file(filename);
    if (!file.eof() && !file.fail()) {
        file.seekg(0, std::ios_base::end);
        std::streampos filesize = file.tellg();
        raw.resize(filesize);

        file.seekg(0, std::ios_base::beg);
        file.read(&raw[0], filesize);
    }

    // decompress in memory
    boost::iostreams::filtering_ostream os;
    os.push(boost::iostreams::gzip_decompressor());
    os.push(boost::iostreams::back_inserter(*rep));
    boost::iostreams::write(os, &raw[0], raw.size());
}

void MNISTLoader::write_img_to_png(unsigned int imgid, const std::string& filename) {
    static const unsigned int imgsz = 28;

    std::vector<uint8_t> data(imgsz * imgsz, 0);

    for(unsigned int i=0; i<(imgsz * imgsz); i++) {
        data[i] = 255 - (uint8_t)this->trainingset[imgid * 784 + 16 + i];
    }

    PNG::write_image_buffer_to_png(filename, data, imgsz, imgsz, PNG_COLOR_TYPE_GRAY);
}

std::shared_ptr<Dataset> MNISTLoader::get_trainingset() const {
    auto dataset = std::make_shared<Dataset>(this->trainingset_size, 784, 10);

    for(unsigned int i=0; i<this->trainingset_size; i++) {
        std::vector<double> in(784);
        std::vector<double> out(10, 0.0);

        for(unsigned int j=0; j<784; j++) {
            in[j] = (double)this->trainingset[i * 784 + 16 + j] / 255.0f;
        }
        out[this->traininglabels[i + 8]] = 1.0;

        dataset->set_input_vector(i, in);
        dataset->set_output_vector(i, out);
    }

    return dataset;
}

std::shared_ptr<Dataset> MNISTLoader::get_testset() const {
    auto dataset = std::make_shared<Dataset>(this->testset_size, 784, 10);

    for(unsigned int i=0; i<this->testset_size; i++) {
        std::vector<double> in(784);
        std::vector<double> out(10, 0.0);

        for(unsigned int j=0; j<784; j++) {
            in[j] = (double)this->testset[i * 784 + 16 + j] / 255.0f;
        }
        out[this->testlabels[i + 8]] = 1.0;

        dataset->set_input_vector(i, in);
        dataset->set_output_vector(i, out);
    }

    return dataset;
}
