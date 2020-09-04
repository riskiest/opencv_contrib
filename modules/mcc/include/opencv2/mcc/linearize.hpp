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

#ifndef __OPENCV_MCC_LINEARIZE_HPP__
#define __OPENCV_MCC_LINEARIZE_HPP__

#include "color.hpp"

namespace cv
{
namespace ccm
{

/* *\ brief Enum of the possible types of linearization.
*/
enum LINEAR_TYPE
{
    IDENTITY_,
    GAMMA,
    COLORPOLYFIT,
    COLORLOGPOLYFIT,
    GRAYPOLYFIT,
    GRAYLOGPOLYFIT
};

/* *\ brief Polyfit model.
*/
class CV_EXPORTS_W Polyfit
{
public:
    int deg;
    cv::Mat p;

    Polyfit() {};

    /* *\ brief Polyfit method.
    */
    Polyfit(cv::Mat s, cv::Mat d, int deg_);

    virtual ~Polyfit() {};

    cv::Mat operator()(const cv::Mat& inp);

private:
    double fromEW(double x);
};

/* *\ brief Logpolyfit model.
*/
class CV_EXPORTS_W LogPolyfit
{
public:
    int deg;
    Polyfit p;

    LogPolyfit() {};

    /* *\ brief Logpolyfit method.
    */
    LogPolyfit(cv::Mat s, cv::Mat d, int deg_);

    virtual ~LogPolyfit() {};

    cv::Mat operator()(const cv::Mat& inp);
};

/* *\ brief Linearization base.
*/
class CV_EXPORTS_W Linear
{
public:
    Linear() {};

    virtual ~Linear() {};

    /* *\ brief Inference.
       *\ param inp the input array, type of cv::Mat.
    */
    virtual cv::Mat linearize(cv::Mat inp);

    /* *\brief Evaluate linearization model.
    */
    virtual void value(void) {};
};

/* *\ brief Linearization identity.
   *        make no change.
*/
class CV_EXPORTS_W LinearIdentity : public Linear {};

/* *\ brief Linearization gamma correction.
*/
class CV_EXPORTS_W LinearGamma : public Linear
{
public:
    double gamma;

    LinearGamma(double gamma_) :gamma(gamma_) {};

    cv::Mat linearize(cv::Mat inp) CV_OVERRIDE;
};

/* *\ brief Linearization.
   *        Grayscale polynomial fitting.
*/
template <class T>
class LinearGray :public Linear
{
public:
    int deg;
    T p;

    LinearGray(int deg_, cv::Mat src, Color dst, cv::Mat mask, RGBBase_ cs) :deg(deg_)
    {
        dst.getGray();
        Mat lear_gray_mask = mask & dst.grays;

        // the grayscale function is approximate for src is in relative color space.
        src = rgb2gray(maskCopyTo(src, lear_gray_mask));
        cv::Mat dst_ = maskCopyTo(dst.toGray(cs.io), lear_gray_mask);
        calc(src, dst_);
    }

    /* *\ brief monotonically increase is not guaranteed.
       *\ param src the input array, type of cv::Mat.
       *\ param dst the input array, type of cv::Mat.
    */
    void calc(const cv::Mat& src, const cv::Mat& dst)
    {
        p = T(src, dst, deg);
    };

    cv::Mat linearize(cv::Mat inp) CV_OVERRIDE
    {
        return p(inp);
    };
};

/* *\ brief Linearization.
   *        Fitting channels respectively.
*/
template <class T>
class LinearColor :public Linear
{
public:
    int deg;
    T pr;
    T pg;
    T pb;

    LinearColor(int deg_, cv::Mat src_, Color dst, cv::Mat mask, RGBBase_ cs) :deg(deg_)
    {
        Mat src = maskCopyTo(src_, mask);
        cv::Mat dst_ = maskCopyTo(dst.to(*cs.l).colors, mask);
        calc(src, dst_);
    }

    void calc(const cv::Mat& src, const cv::Mat& dst)
    {
        cv::Mat schannels[3];
        cv::Mat dchannels[3];
        split(src, schannels);
        split(dst, dchannels);
        pr = T(schannels[0], dchannels[0], deg);
        pg = T(schannels[1], dchannels[1], deg);
        pb = T(schannels[2], dchannels[2], deg);
    };

    cv::Mat linearize(cv::Mat inp) CV_OVERRIDE
    {
        cv::Mat channels[3];
        split(inp, channels);
        std::vector<cv::Mat> channel;
        cv::Mat res;
        merge(std::vector<cv::Mat>{ pr(channels[0]), pr(channels[1]), pr(channels[2]) }, res);
        return res;
    };
};


/* *\ brief Get linearization method.
   *        used in ccm model.
   *\ param gamma used in LinearGamma.
   *\ param deg degrees.
   *\ param src the input array, type of cv::Mat.
   *\ param dst the input array, type of cv::Mat.
   *\ param mask the input array, type of cv::Mat.
   *\ param cs type of RGBBase_.
   *\ param linear_type type of linear.
*/
std::shared_ptr<Linear> getLinear(double gamma, int deg, cv::Mat src, Color dst, cv::Mat mask, RGBBase_ cs, LINEAR_TYPE linear_type);

} // namespace ccm
} // namespace cv


#endif
