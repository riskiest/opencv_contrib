// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * MIT License
 *
 * Copyright (c) 2018 Pedro Diamel Marrero Fernández
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef __OPENCV_MCC_OPERATIONS_HPP__
#define __OPENCV_MCC_OPERATIONS_HPP__

#include <functional>
#include <vector>
#include "utils.hpp"

namespace cv
{
namespace ccm
{

typedef std::function<cv::Mat(cv::Mat)> MatFunc;

/* *\ brief Operation class contains some operarions used for color space
   *        conversion containing linear transformation and non-linear transformation
 */
class CV_EXPORTS Operation
{
public:
    CV_PROP_RW bool linear;
    CV_PROP_RW cv::Mat M;
    CV_PROP_RW MatFunc f;

    Operation() : linear(true), M(cv::Mat()) {};

    Operation(cv::Mat M_) :linear(true), M( M_ ) {};

    Operation(MatFunc f_) : linear(false), f(f_) {};

    virtual ~Operation() {};

    /* *\ brief operator function will run operation*/
    cv::Mat operator()(cv::Mat& abc);

    /* *\ brief add function will conbine this operation
     *          with  other  linear transformation operation*/
    CV_WRAP void add(const Operation& other);

    CV_WRAP void clear();
};

const Operation IDENTITY_OP( [](cv::Mat x) {return x; } );

class CV_EXPORTS Operations
{
public:
    CV_PROP_RW std::vector<Operation> ops;

    Operations() :ops{ } {};

    Operations(std::initializer_list<Operation> op) :ops{ op } {};

    virtual ~Operations() {};

    /* *\ brief add function will conbine this operation with  other  transformation operations*/
    Operations& add(const Operations& other);

    /* *\ brief run operations to make color conversion*/
    CV_WRAP cv::Mat run(cv::Mat abc);
};

const Operations IDENTITY_OPS{ IDENTITY_OP };

} // namespace ccm
} // namespace cv


#endif