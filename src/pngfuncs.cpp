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

#include "pngfuncs.h"

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

/**
 * @brief      Writes an image buffer to png.
 *
 * @param[in]  filename  The filename
 * @param[in]  buffer    The buffer
 * @param[in]  width     The width
 * @param[in]  height    The height
 * @param[in]  col       The color type
 */
void PNG::write_image_buffer_to_png(const std::string& filename, const std::vector<uint8_t>& buffer, unsigned int width, unsigned int height, unsigned int col) {
    png_structp png_ptr;
    png_infop info_ptr;

    /* create file */
    std::ofstream ofile(filename.c_str(), std::ios::binary);

    if (!ofile.is_open() ) {
        std::cerr << "[write_png_file] File " << filename.c_str() << " could not be opened for reading" << std::endl;
        exit(-1);
    }

    /* initialize stuff */
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr) {
        std::cerr << "[write_png_file] png_create_write_struct failed" << std::endl;
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        std::cerr << "[write_png_file] png_create_info_struct failed" << std::endl;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        std::cerr << "[write_png_file] Error during init_io";
    }

    png_set_write_fn(png_ptr, (void *)&ofile, write_file_callback, NULL);

    /* write header */
    if (setjmp(png_jmpbuf(png_ptr))) {
        std::cerr << "[write_png_file] Error during writing header" << std::endl;
    }

    png_set_IHDR(png_ptr, info_ptr, width, height,
                 8, col, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);


    /* write bytes */
    if (setjmp(png_jmpbuf(png_ptr))) {
        std::cerr << "[write_png_file] Error during writing bytes" << std::endl;
    }

    if(!(col == PNG_COLOR_TYPE_GRAY | col == PNG_COLOR_TYPE_RGBA)) {
        throw std::runtime_error("Unsupported color type requested.");
    }

    png_bytep *row_pointers;
    if(col == PNG_COLOR_TYPE_GRAY) {
        row_pointers = new png_bytep[height];
        for(unsigned int i=0; i<height; i++) {
            row_pointers[i] = new unsigned char[width];
            for(unsigned int j=0; j<width; j++) {
                row_pointers[i][j] = buffer[i * width + j];
            }
        }
    } else {
        static const unsigned int coldepth = 4;
        row_pointers = new png_bytep[height];
        for(unsigned int i=0; i<height; i++) {
            row_pointers[i] = new unsigned char[width * coldepth];
            for(unsigned int j=0; j<width; j++) {
                for(unsigned int p=0; p<coldepth; p++) {
                    // note that height needs to inverted for correct image capture
                    row_pointers[i][j * coldepth + p] = buffer[((height - i - 1) * width + j) * coldepth + p];
                }
            }
        }
    }
    png_write_image(png_ptr, row_pointers);

    /* end write */
    if (setjmp(png_jmpbuf(png_ptr))) {
        std::cerr << "[write_png_file] Error during end of write" << std::endl;
    }

    png_write_end(png_ptr, NULL);

    for(unsigned int i=0; i<height; i++) {
        delete[] row_pointers[i];
    }
    delete[] row_pointers;

    png_destroy_write_struct(&png_ptr, &info_ptr);

    ofile.close();
}

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
void PNG::load_image_buffer_from_png(const std::string& filename, std::vector<uint8_t>& buffer, png_uint_32* width, png_uint_32* height, int* col, int* bit_depth) {
    png_structp png_ptr;
    png_infop info_ptr;

    char header[8];    // 8 is the maximum size that can be checked

    /* open file and test for it being a png */
    std::ifstream ifile(filename.c_str(), std::ios::binary);

    if (!ifile.is_open() ) {
        std::cerr << "[read_png_file] File " << filename.c_str() << " could not be opened for reading" << std::endl;
        exit(-1);
    }

    ifile.read(&header[0], 8 * sizeof(char));
    if (!png_check_sig((png_bytep)header, 8)) {
        std::cerr << "[read_png_file] File " << filename.c_str() << " is not recognized as a PNG file" << std::endl;
        exit(-1);
    }

    /* initialize stuff */
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);

    if (!png_ptr) {
        std::cerr << "[read_png_file] png_create_read_struct failed for file " << filename.c_str() << std::endl;
        exit(-1);
    }

    info_ptr = png_create_info_struct(png_ptr);

    if (!info_ptr) {
        std::cerr << "[read_png_file] png_create_info_struct failed for file " << filename.c_str() << std::endl;
        exit(-1);
    }

    png_set_read_fn(png_ptr, (void*)&ifile, read_file_callback);

    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    png_get_IHDR(png_ptr, info_ptr, (png_uint_32*)width, (png_uint_32*)height, bit_depth, col, 0, 0, 0);

    // allocate space for storing texture
    png_bytepp row_pointers = new png_bytep[(*height)];
    for (unsigned int y=0; y<(*height); y++) {
        row_pointers[y] = new png_byte[png_get_rowbytes(png_ptr,info_ptr)];
    }

    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, info_ptr);

    // transfer data to std::vector
    buffer.resize((*width) * (*height), 0);
    for(unsigned int i=0; i<(*height); i++) {
        for(unsigned int j=0; j<(*width); j++) {
            buffer[i * (*width) + j] = row_pointers[i][j];
        }
    }

    ifile.close();

    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

    for(unsigned int i=0; i<*height; i++) {
        delete[] row_pointers[i];
    }
    delete[] row_pointers;
}

void PNG::read_file_callback( png_structp png_ptr, png_bytep out, png_size_t count ) {
    png_voidp io_ptr = png_get_io_ptr( png_ptr );

    if( io_ptr == 0 ) {
        return;
    }

    std::ifstream &ifs = *(std::ifstream*)io_ptr;

    ifs.read( (char*)out, count );
}

void PNG::write_file_callback(png_structp png_ptr, png_bytep data, png_size_t count) {
    std::ofstream *outfile = (std::ofstream*)png_get_io_ptr(png_ptr);
    outfile->write((char*)data, count);
}
