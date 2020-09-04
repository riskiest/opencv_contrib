// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef __OPENCV_MCC_UTILS_HPP__
#define __OPENCV_MCC_UTILS_HPP__

#include <opencv2/core.hpp>


namespace cv {
namespace ccm {

/* *\ brief gamma correction ,see ColorSpace.pdf for details.
 *\ param src the input array,type of cv::Mat.
 *\ param gamma a constant for gamma correction.
 */
CV_EXPORTS double gammaCorrection_(const double& element, const double& gamma);

CV_EXPORTS cv::Mat gammaCorrection(const cv::Mat& src, const double& gamma);

/* *\ brief maskCopyTo a function to delete unsatisfied elementwise.
 *\ param src the input array, type of cv::Mat.
 *\ param mask operation mask that used to choose satisfided elementwise.
 */
CV_EXPORTS cv::Mat maskCopyTo(const cv::Mat& src, const cv::Mat& mask);

/* *\ brief multiple the function used to compute an array with n channels
 *mulipied by ccm. \ param src the input array, type of cv::Mat. \ param ccm the
 *ccm matrix to make color correction.
 */
CV_EXPORTS cv::Mat multiple(const cv::Mat& xyz, const cv::Mat& ccm);

/* *\ brief multiple the function used to get the mask of saturated colors,
            colors between low and up will be choosed.
   *\ param src the input array, type of cv::Mat.
   *\ param low  the threshold to choose saturated colors
   *\ param up  the threshold to choose saturated colors
*/
CV_EXPORTS cv::Mat saturate(cv::Mat& src, const double& low, const double& up);

/* *\ brief rgb2gray it is an approximation grayscale function for relative RGB
 *color space, see Miscellaneous.pdf for details; \ param rgb the input array,
 *type of cv::Mat.
 */
CV_EXPORTS cv::Mat rgb2gray(cv::Mat rgb);

/* *\ brief function for elementWise operation
 *\ param src the input array, type of cv::Mat
 *\ lambda a for operation
 */
template<typename F>
cv::Mat elementWise(const cv::Mat& src, F&& lambda)
{
    cv::Mat dst = src.clone();
    const int channel = src.channels();
    switch (channel)
    {
    case 1:
    {
        cv::MatIterator_<double> it, end;
        for (it = dst.begin<double>(), end = dst.end<double>(); it != end; ++it)
        {
            (*it) = lambda((*it));
        }
        break;
    }
    case 3:
    {
        cv::MatIterator_<cv::Vec3d> it, end;
        for (it = dst.begin<cv::Vec3d>(), end = dst.end<cv::Vec3d>(); it != end; ++it)
        {
            for (int j = 0; j < 3; j++)
            {
                (*it)[j] = lambda((*it)[j]);
            }
        }
        break;
    }
    default:
        throw std::invalid_argument { "Wrong channel!" };
        break;
    }
    return dst;
}

/* *\ brief function for channel operation
   *\ param src the input array, type of cv::Mat
   *\ lambda the function for operation
*/
template<typename F>
cv::Mat channelWise(const cv::Mat& src, F&& lambda)
{
    cv::Mat dst = src.clone();
    cv::MatIterator_<cv::Vec3d> it, end;
    for (it = dst.begin<cv::Vec3d>(), end = dst.end<cv::Vec3d>(); it != end; ++it)
    {
        *it = lambda(*it);
    }
    return dst;
}

/* *\ brief function for distance operation.
 *\ param src the input array, type of cv::Mat.
 *\ param ref another input array, type of cv::Mat.
 *\ param lambda the computing method for distance .
 */
template<typename F>
cv::Mat distanceWise(cv::Mat& src, cv::Mat& ref, F&& lambda)
{
    cv::Mat dst = cv::Mat(src.size(), CV_64FC1);
    cv::MatIterator_<cv::Vec3d> it_src = src.begin<cv::Vec3d>(), end_src = src.end<cv::Vec3d>(),
        it_ref = ref.begin<cv::Vec3d>();
    cv::MatIterator_<double> it_dst = dst.begin<double>();
    for (; it_src != end_src; ++it_src, ++it_ref, ++it_dst)
    {
        *it_dst = lambda(*it_src, *it_ref);
    }
    return dst;
}

CV_EXPORTS cv::Mat multiple(const cv::Mat& xyz, const cv::Mat& ccm);

const static cv::Mat m_gray = (cv::Mat_<double>(3, 1) << 0.2126, 0.7152, 0.0722);

}  // namespace ccm
}  // namespace cv

#endif