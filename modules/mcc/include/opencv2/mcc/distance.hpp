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

#ifndef __OPENCV_MCC_DISTANCE_HPP__
#define __OPENCV_MCC_DISTANCE_HPP__

#include "utils.hpp"

namespace cv {
namespace ccm {

/* *\brief Enum of possibale functions to calculate the distance between
   *       colors.see https://en.wikipedia.org/wiki/Color_difference for details;*/
enum DISTANCE_TYPE {
    CIE76,
    CIE94_GRAPHIC_ARTS,
    CIE94_TEXTILES,
    CIE2000,
    CMC_1TO1,
    CMC_2TO1,
    RGB,
    RGBL
};

/* *\ brief  distance between two points in formula CIE76
   *\ param lab1 a 3D vector
   *\ param lab2 a 3D vector
   *\ return distance between lab1 and lab2
*/
CV_EXPORTS double deltaCIE76(cv::Vec3d lab1, cv::Vec3d lab2);

/* *\ brief  distance between two points in formula CIE94
   *\ param lab1 a 3D vector
   *\ param lab2 a 3D vector
   *\ param kH Hue scale
   *\ param kC Chroma scale
   *\ param kL Lightness scale
   *\ param k1 first scale parameter
   *\ param k2 second scale parameter
   *\ return distance between lab1 and lab2
*/
CV_EXPORTS double deltaCIE94(cv::Vec3d lab1, cv::Vec3d lab2, double kH = 1.0,
                  double kC = 1.0, double kL = 1.0, double k1 = 0.045,
                  double k2 = 0.015);

CV_EXPORTS double deltaCIE94GraphicArts(cv::Vec3d lab1, cv::Vec3d lab2);

CV_EXPORTS double toRad(double degree);

CV_EXPORTS double deltaCIE94Textiles(cv::Vec3d lab1, cv::Vec3d lab2);

/* *\ brief  distance between two points in formula CIE2000
   *\ param lab1 a 3D vector
   *\ param lab2 a 3D vector
   *\ param kL Lightness scale
   *\ param kC Chroma scale
   *\ param kH Hue scale
   *\ return distance between lab1 and lab2
*/
CV_EXPORTS double deltaCIEDE2000_(cv::Vec3d lab1, cv::Vec3d lab2, double kL = 1.0,
                       double kC = 1.0, double kH = 1.0);

CV_EXPORTS double deltaCIEDE2000(cv::Vec3d lab1, cv::Vec3d lab2);

/* *\ brief  distance between two points in formula CMC
   *\ param lab1 a 3D vector
   *\ param lab2 a 3D vector
   *\ param kL Lightness scale
   *\ param kC Chroma scale
   *\ return distance between lab1 and lab2
*/
CV_EXPORTS double deltaCMC(cv::Vec3d lab1, cv::Vec3d lab2, double kL = 1, double kC = 1);

CV_EXPORTS double deltaCMC1To1(cv::Vec3d lab1, cv::Vec3d lab2);

CV_EXPORTS double deltaCMC2To1(cv::Vec3d lab1, cv::Vec3d lab2);

CV_EXPORTS cv::Mat distance(cv::Mat src, cv::Mat ref, DISTANCE_TYPE distance_type);

}  // namespace ccm
}  // namespace cv

#endif