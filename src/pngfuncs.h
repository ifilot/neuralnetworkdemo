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

#ifndef _UTIL_PNG_H
#define _UTIL_PNG_H

#include <fstream>
#include <iostream>
#include <vector>
#include <png.h>

/*
 * Routines to write a color buffer to a png file
 *
 * Color types:
 * ============
 * - PNG_COLOR_TYPE_GRAY
 * - PNG_COLOR_TYPE_GRAY_ALPHA
 * - PNG_COLOR_TYPE_PALETTE
 * - PNG_COLOR_TYPE_RGB
 * - PNG_COLOR_TYPE_RGB_ALPHA
 *
 * Bit depths:
 * ===========
 * - 8 or 16 bits, except for:
 *      - PNG_COLOR_TYPE_GRAY: 1, 2, 4, 8, or 16 bits
 *      - PNG_COLOR_TYPE_PALETTE: 1, 2, 4, 8
 */

namespace PNG {

    /**
     * @brief      Writes an image buffer to png.
     *
     * @param[in]  filename  The filename
     * @param[in]  buffer    The buffer
     * @param[in]  width     The width
     * @param[in]  height    The height
     * @param[in]  col       The color type
     */
    void write_image_buffer_to_png(const std::string& filename, const std::vector<uint8_t>& buffer, unsigned int width, unsigned int height, unsigned int col);

    /**
     * @brief      Loads an image buffer from png.
     *
     * @param[in]  filename   The filename
     * @param      buffer     The buffer
     * @param      width      The width
     * @param      height     The height
     * @param      col        The color type
     * @param      bit_depth  The bit depth
     *
     */
    void load_image_buffer_from_png(const std::string& filename, std::vector<uint8_t>& buffer, png_uint_32* width, png_uint_32* height, int* col, int* bit_depth);

    void read_file_callback(png_structp png_ptr, png_bytep out, png_size_t count);

    void write_file_callback(png_structp png_ptr, png_bytep data, png_size_t count);
}

#endif //_UTIL_PNG_H
