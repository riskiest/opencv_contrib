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

#include "precomp.hpp"

namespace cv
{
namespace ccm
{

Polyfit::Polyfit(cv::Mat s, cv::Mat d, int deg_) :deg(deg_)
{
    int npoints = s.checkVector(1);
    cv::Mat_<double> srcX(s), srcY(d);
    cv::Mat_<double> m = cv::Mat_<double>::ones(npoints, deg + 1);
    for (int y = 0; y < npoints; ++y)
    {
        for (int x = 1; x < m.cols; ++x)
        {
            m.at<double>(y, x) = srcX.at<double>(y) * m.at<double>(y, x -1);
        }
    }
    cv::solve(m, srcY, p, DECOMP_SVD);
}

cv::Mat Polyfit::operator()(const cv::Mat& inp)
{
    return elementWise(inp, [this](double a)->double {return fromEW(a); });
};

double Polyfit::fromEW(double x)
{
    double res = 0;
    for (int d = 0; d <= deg; ++d)
    {
        res += pow(x, d) * p.at<double>(d, 0);
    }
    return res;
};

LogPolyfit::LogPolyfit(cv::Mat s, cv::Mat d, int deg_) :deg(deg_)
{
    cv::Mat mask_ = (s > 0) & (d > 0);
    cv::Mat src_, dst_, s_, d_;
    src_ = maskCopyTo(s, mask_);
    dst_ = maskCopyTo(d, mask_);
    log(src_, s_);
    log(dst_, d_);
    p = Polyfit(s_, d_, deg);
}

cv::Mat LogPolyfit::operator()(const cv::Mat& inp)
{
    cv::Mat mask_ = inp >= 0;
    cv::Mat y, y_, res;
    log(inp, y);
    y = p(y);
    exp(y, y_);
    y_.copyTo(res, mask_);
    return res;
};

cv::Mat Linear::linearize(cv::Mat inp)
{
    return inp;
};

cv::Mat LinearGamma::linearize(cv::Mat inp)
{
    return gammaCorrection(inp, gamma);
};

std::shared_ptr<Linear> getLinear(double gamma, int deg, cv::Mat src, Color dst, cv::Mat mask, RGBBase_ cs, LINEAR_TYPE linear_type)
{
    std::shared_ptr<Linear> p = std::make_shared<Linear>();
    switch (linear_type)
    {
    case cv::ccm::IDENTITY_:
        p.reset(new LinearIdentity());
        break;
    case cv::ccm::GAMMA:
        p.reset(new LinearGamma(gamma));
        break;
    case cv::ccm::COLORPOLYFIT:
        p.reset(new LinearColor<Polyfit>(deg, src, dst, mask, cs));
        break;
    case cv::ccm::COLORLOGPOLYFIT:
        p.reset(new LinearColor<LogPolyfit>(deg, src, dst, mask, cs));
        break;
    case cv::ccm::GRAYPOLYFIT:
        p.reset(new LinearGray<Polyfit>(deg, src, dst, mask, cs));
        break;
    case cv::ccm::GRAYLOGPOLYFIT:
        p.reset(new LinearGray<LogPolyfit>(deg, src, dst, mask, cs));
        break;
    default:
        throw std::invalid_argument{ "Wrong linear_type!" };
        break;
    }
    return p;
};

} // namespace ccm
} // namespace cv