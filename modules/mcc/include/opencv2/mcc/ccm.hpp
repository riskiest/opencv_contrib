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

#ifndef __OPENCV_MCC_CCM_HPP__
#define __OPENCV_MCC_CCM_HPP__

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "linearize.hpp"

namespace cv
{
namespace ccm
{

/**
    src :
            detected colors of ColorChecker patches;
            NOTICE: the color type is RGB not BGR, and the color values are in [0, 1];
            type: cv::Mat;
    dst :
            the reference colors;
            NOTICE: Built-in color card or custom color card are supported;
                    Built-in:
                        Macbeth_D50_2: Macbeth ColorChecker with 2deg D50;
                        Macbeth_D65_2: Macbeth ColorChecker with 2deg D65;
                    Custom:
                        You should use Color
                        For the list of color spaces supported, see the notes below;
                        If the color type is some RGB, the format is RGB not BGR, and the color values are in [0, 1];
            type: Color;
    colorspace :
            the absolute color space that detected colors convert to;
            NOTICE: it should be some RGB color space;
                    For the list of RGB color spaces supported, see the notes below;
            type: ColorSpace;
    ccm_type :
            the shape of color correction matrix(CCM);
            Supported list:
                "CCM_3x3": 3x3 matrix;
                "CCM_4x3": 4x3 matrix;
            type: enum CCM_TYPE;
            default: CCM_3x3;
    distance :
            the type of color distance;
            Supported list:
                "CIE2000";
                "CIE94_GRAPHIC_ARTS";
                "CIE94_TEXTILES";
                "CIE76";
                "CMC_1TO1";
                "CMC_2TO1";
                "RGB" : Euclidean distance of rgb color space;
                "RGBL" : Euclidean distance of rgbl color space;
            type: enum DISTANCE_TYPE;
            default: CIE2000;
    linear_type :
            the method of linearization;
            NOTICE: see Linearization.pdf for details;
            Supported list:
                "IDENTITY_" : no change is made;
                "GAMMA": gamma correction;
                        Need assign a value to gamma simultaneously;
                "COLORPOLYFIT": polynomial fitting channels respectively;
                                Need assign a value to deg simultaneously;
                "GRAYPOLYFIT": grayscale polynomial fitting;
                                Need assign a value to deg and dst_whites simultaneously;
                "COLORLOGPOLYFIT": logarithmic polynomial fitting channels respectively;
                                Need assign a value to deg simultaneously;
                "GRAYLOGPOLYFIT": grayscale Logarithmic polynomial fitting;
                                Need assign a value to deg and dst_whites simultaneously;
            type: enum LINEAR_TYPE;
            default: IDENTITY_;
    gamma :
            the gamma value of gamma correction;
            NOTICE: only valid when linear is set to "gamma";
            type: double;
            default: 2.2;
    deg :
            the degree of linearization polynomial;
            NOTICE: only valid when linear is set to "COLORPOLYFIT", "GRAYPOLYFIT",
                    "COLORLOGPOLYFIT" and "GRAYLOGPOLYFIT";
            type: int;
            default: 3;
    saturated_threshold :
            the threshold to determine saturation;
            NOTICE: it is a tuple of [low, up];
                    The colors in the closed interval [low, up] are reserved to participate
                    in the calculation of the loss function and initialization parameters.
            type: std::vector<double>;
            default: { 0, 0.98 };
    ---------------------------------------------------
    There are some ways to set weights:
        1. set weights_list only;
        2. set weights_coeff only;
    see CCM.pdf for details;

    weights_list :
            the list of weight of each color;
            type: cv::Mat;
            default: empty array;
    weights_coeff :
            the exponent number of L* component of the reference color in CIE Lab color space;
            type: double;
            default: 0;
    ---------------------------------------------------
    initial_method_type :
            the method of calculating CCM initial value;
            see CCM.pdf for details;
            Supported list:
                'LEAST_SQUARE': least-squre method;
                'WHITE_BALANCE': white balance method;
            type: enum INITIAL_METHOD_TYPE;
    max_count, epsilon :
            used in MinProblemSolver-DownhillSolver;
            Terminal criteria to the algorithm;
            type: int, double;
            default: 5000, 1e-4;


    ---------------------------------------------------
    Supported Color Space:
            Supported list of RGB color spaces:
                sRGB;
                AdobeRGB;
                WideGamutRGB;
                ProPhotoRGB;
                DCI_P3_RGB;
                AppleRGB;
                REC_709_RGB;
                REC_2020_RGB;

            Supported list of linear RGB color spaces:
                sRGBL;
                AdobeRGBL;
                WideGamutRGBL;
                ProPhotoRGBL;
                DCI_P3_RGBL;
                AppleRGBL;
                REC_709_RGBL;
                REC_2020_RGBL;

            Supported list of non-RGB color spaces:
                Lab_D50_2;
                Lab_D65_2;
                XYZ_D50_2;
                XYZ_D65_2;

            Supported IO (You can use Lab(io) or XYZ(io) to create color space):
                A_2;
                A_10;
                D50_2;
                D50_10;
                D55_2;
                D55_10;
                D65_2;
                D65_10;
                D75_2;
                D75_10;
                E_2;
                E_10;


    ---------------------------------------------------
    Abbr.
        src, s: source;
        dst, d: destination;
        io: illuminant & observer;
        sio, dio: source of io; destination of io;
        rgbl: linear RGB
        cs: color space;
        cc: Colorchecker;
        M, m: matrix
        ccm: color correction matrix;
        cam: chromatic adaption matrix;

*/


/* *\ brief Enum of the possible types of ccm.
*/
enum CCM_TYPE
{
    CCM_3x3,
    CCM_4x3
};

/* *\ brief Enum of the possible types of initial method.
*/
enum INITIAL_METHOD_TYPE
{
    WHITE_BALANCE,
    LEAST_SQUARE
};


/* *\ brief Core class of ccm model.
   *        produce a ColorCorrectionModel instance for inference.
*/
class CV_EXPORTS ColorCorrectionModel
{
public:
    // detected colors, the referenceand the RGB colorspace for conversion
    CV_PROP_RW cv::Mat src;
    CV_PROP_RW Color dst;
    CV_PROP_RW RGBBase_ &cs;

    // ccm type and shape
    CV_PROP_RW CCM_TYPE ccm_type;
    CV_PROP_RW int shape;

    // linear method and distance
    CV_PROP_RW std::shared_ptr<Linear> linear;
    CV_PROP_RW DISTANCE_TYPE distance;

    CV_PROP_RW cv::Mat weights;
    CV_PROP_RW cv::Mat ccm;
    CV_PROP_RW cv::Mat ccm0;

    CV_PROP_RW int max_count;
    CV_PROP_RW double epsilon;


    ColorCorrectionModel(cv::Mat src_, Color dst_, RGBBase_ &cs_, CCM_TYPE ccm_type_, DISTANCE_TYPE distance_, LINEAR_TYPE linear_type,
                         double gamma, int deg, std::vector<double> saturated_threshold, cv::Mat weights_list, double weights_coeff,
                         INITIAL_METHOD_TYPE initial_method_type, int max_count_, double epsilon_);

    /* *\ brief Make no change for CCM_3x3.
       *        convert cv::Mat A to [A, 1] in CCM_4x3.
       *\ param inp the input array, type of cv::Mat.
       *\ return the output array, type of cv::Mat
    */
    CV_WRAP cv::Mat prepare(const cv::Mat &inp);

    /* *\ brief Fitting nonlinear - optimization initial value by white balance.
       *        see CCM.pdf for details.
       *\ return the output array, type of cv::Mat
    */
    CV_WRAP cv::Mat initialWhiteBalance(void);

    /* *\ brief Fitting nonlinear-optimization initial value by least square.
       *        see CCM.pdf for details
       *\ param fit if fit is True, return optimalization for rgbl distance function.
    */
    CV_WRAP void initialLeastSquare(bool fit = false);

    /* *\ brief Loss function base on cv::MinProblemSolver::Function.
       *        see details in https://github.com/opencv/opencv/blob/master/modules/core/include/opencv2/core/optim.hpp
    */
    class LossFunction : public cv::MinProblemSolver::Function
    {
    public:
        ColorCorrectionModel *ccm_loss;
        LossFunction(ColorCorrectionModel *ccm) : ccm_loss(ccm){};

        /* *\ brief Reset dims to ccm->shape.
        */
        int getDims() const CV_OVERRIDE
        {
            return ccm_loss->shape;
        }

        /* *\ brief Reset calculation.
        */
        double calc(const double *x) const CV_OVERRIDE
        {
            cv::Mat ccm(ccm_loss->shape, 1, CV_64F);
            for (int i = 0; i < ccm_loss->shape; i++)
            {
                ccm.at<double>(i, 0) = x[i];
            }
            ccm = ccm.reshape(0, ccm_loss->shape / 3);
            Mat reshapecolor = ccm_loss->src_rgbl.reshape(1, 0) * ccm;
            cv::Mat dist = Color(reshapecolor.reshape(3, 0), ccm_loss->cs).diff(ccm_loss->dst, ccm_loss->distance);
            cv::Mat dist_;
            pow(dist, 2, dist_);
            if (!ccm_loss->weights.empty())
            {
                dist_ = ccm_loss->weights.mul(dist_);
            }
            Scalar ss = sum(dist_);
            return ss[0];
        }
    };

    /* *\ brief Fitting ccm if distance function is associated with CIE Lab color space.
       *        see details in https://github.com/opencv/opencv/blob/master/modules/core/include/opencv2/core/optim.hpp
       *        Set terminal criteria for solver is possible.
    */
    CV_WRAP void fitting(void);

    /* *\ brief Infer using fitting ccm.
       *\ param img the input image, type of cv::Mat.
       *\ return the output array, type of cv::Mat.
    */
    CV_WRAP cv::Mat infer(const cv::Mat &img, bool islinear = false);

    /* *\ brief Infer image and output as an BGR image with uint8 type.
       *        mainly for test or debug.
       *        input size and output size should be 255.
       *\ param imgfile path name of image to infer.
       *\ param islinear if linearize or not.
       *\ return the output array, type of cv::Mat.
    */
    CV_WRAP cv::Mat inferImage(std::string imgfile, bool islinear = false);

private:
    cv::Mat mask;
    cv::Mat dist;
    int masked_len;
    double loss;

    // RGBl of detected data and the reference
    cv::Mat src_rgbl;
    cv::Mat dst_rgbl;

    /* *\ brief Calculate weights and mask.
       *\ param weights_list the input array, type of cv::Mat.
       *\ param weights_coeff type of double.
       *\ param saturate_list the input array, type of cv::Mat.
    */
    void calWeightsMasks(cv::Mat weights_list, double weights_coeff, cv::Mat saturate_mask);

};

} // namespace ccm
} // namespace cv


#endif