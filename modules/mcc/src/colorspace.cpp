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

bool ColorSpace::relate(const ColorSpace& other) const
{
    return (type == other.type) && (io == other.io);
};

Operations ColorSpace::relation(const ColorSpace& /*other*/) const
{
    return IDENTITY_OPS;
};

bool ColorSpace::operator<(const ColorSpace& other)const
{
    return (io < other.io || (io == other.io && type < other.type) || (io == other.io && type == other.type && linear < other.linear));
}

Operations RGBBase_::relation(const ColorSpace& other) const
{
    if (linear == other.linear)
    {
        return IDENTITY_OPS;
    }
    if (linear)
    {
        return Operations({ Operation(fromL) });
    }
    return Operations({ Operation(toL) });
};

void RGBBase_::init()
{
    setParameter();
    calLinear();
    calM();
    calOperations();
}

void RGBBase_::bind(RGBBase_& rgbl)
{
    init();
    rgbl.init();
    l = &rgbl;
    rgbl.l = &rgbl;
    nl = this;
    rgbl.nl = this;
}

void RGBBase_::calM()
{
    cv::Mat XYZr, XYZg, XYZb, XYZ_rgbl, Srgb;
    XYZr = cv::Mat(xyY2XYZ({ xr, yr }), true);
    XYZg = cv::Mat(xyY2XYZ({ xg, yg }), true);
    XYZb = cv::Mat(xyY2XYZ({ xb, yb }), true);
    merge(std::vector<cv::Mat>{ XYZr, XYZg, XYZb }, XYZ_rgbl);
    XYZ_rgbl = XYZ_rgbl.reshape(1, XYZ_rgbl.rows);
    cv::Mat XYZw = cv::Mat(illuminants.find(io)->second, true);
    solve(XYZ_rgbl, XYZw, Srgb);
    merge(std::vector<cv::Mat>{ Srgb.at<double>(0)* XYZr,
        Srgb.at<double>(1)* XYZg,
        Srgb.at<double>(2)* XYZb }, M_to);
    M_to = M_to.reshape(1, M_to.rows);
    M_from = M_to.inv();
};

void RGBBase_::calOperations()
{
    // rgb -> rgbl
    toL = [this](cv::Mat rgb)->cv::Mat {return toLFunc(rgb); };
    // rgbl -> rgb
    fromL = [this](cv::Mat rgbl)->cv::Mat {return fromLFunc(rgbl); };
    if (linear)
    {
        to = Operations({ Operation(M_to.t()) });
        from = Operations({ Operation(M_from.t()) });
    }
    else
    {
        to = Operations({ Operation(toL), Operation(M_to.t()) });
        from = Operations({ Operation(M_from.t()), Operation(fromL) });
    }
}

cv::Mat RGBBase_::toLFunc(cv::Mat& /*rgb*/)
{
    return cv::Mat();
};

cv::Mat RGBBase_::fromLFunc(cv::Mat& /*rgbl*/)
{
    return cv::Mat();
};

cv::Mat AdobeRGBBase_::toLFunc(cv::Mat& rgb)
{
    return gammaCorrection(rgb, gamma);
}

cv::Mat AdobeRGBBase_::fromLFunc(cv::Mat& rgbl)
{
    return gammaCorrection(rgbl, 1. / gamma);
}

void sRGBBase_::calLinear()
{
    alpha = a + 1;
    K0 = a / (gamma - 1);
    phi = (pow(alpha, gamma) * pow(gamma - 1, gamma - 1)) / (pow(a, gamma - 1) * pow(gamma, gamma));
    beta = K0 / phi;
}

double sRGBBase_::toLFuncEW(double& x)
{
    if (x > K0)
    {
        return pow(((x + alpha - 1) / alpha), gamma);
    }
    else if (x >= -K0)
    {
        return x / phi;
    }
    else
    {
        return -(pow(((-x + alpha - 1) / alpha), gamma));
    }
}

cv::Mat sRGBBase_::toLFunc(cv::Mat& rgb)
{
    return elementWise(rgb, [this](double a_)->double {return toLFuncEW(a_); });
}

double sRGBBase_::fromLFuncEW(double& x)
{
    if (x > beta)
    {
        return alpha * pow(x, 1 / gamma) - (alpha - 1);
    }
    else if (x >= -beta)
    {
        return x * phi;
    }
    else
    {
        return -(alpha * pow(-x, 1 / gamma) - (alpha - 1));
    }
}

cv::Mat sRGBBase_::fromLFunc(cv::Mat& rgbl)
{
    return elementWise(rgbl, [this](double a_)->double {return fromLFuncEW(a_); });
}

void sRGB_::setParameter()
{
    xr = 0.64;
    yr = 0.33;
    xg = 0.3;
    yg = 0.6;
    xb = 0.15;
    yb = 0.06;
    a = 0.055;
    gamma = 2.4;
}

void AdobeRGB_::setParameter()
{
    xr = 0.64;
    yr = 0.33;
    xg = 0.21;
    yg = 0.71;
    xb = 0.15;
    yb = 0.06;
    gamma = 2.2;
}

void WideGamutRGB_::setParameter()
{
    xr = 0.7347;
    yr = 0.2653;
    xg = 0.1152;
    yg = 0.8264;
    xb = 0.1566;
    yb = 0.0177;
    gamma = 2.2;
}

void ProPhotoRGB_::setParameter()
{
    xr = 0.734699;
    yr = 0.265301;
    xg = 0.159597;
    yg = 0.840403;
    xb = 0.036598;
    yb = 0.000105;
    gamma = 1.8;
}

void DCI_P3_RGB_::setParameter()
{
    xr = 0.68;
    yr = 0.32;
    xg = 0.265;
    yg = 0.69;
    xb = 0.15;
    yb = 0.06;
    gamma = 2.2;
}

void AppleRGB_::setParameter()
{
    xr = 0.625;
    yr = 0.34;
    xg = 0.28;
    yg = 0.595;
    xb = 0.155;
    yb = 0.07;
    gamma = 1.8;
}

void REC_709_RGB_::setParameter()
{
    xr = 0.64;
    yr = 0.33;
    xg = 0.3;
    yg = 0.6;
    xb = 0.15;
    yb = 0.06;
    a = 0.099;
    gamma = 1 / 0.45;
}

void REC_2020_RGB_::setParameter()
{
    xr = 0.708;
    yr = 0.292;
    xg = 0.17;
    yg = 0.797;
    xb = 0.131;
    yb = 0.046;
    a = 0.09929682680944;
    gamma = 1 / 0.45;
}


sRGB_ sRGB = sRGB_(false);
sRGB_ sRGBL = sRGB_(true);
AdobeRGB_ AdobeRGB = AdobeRGB_(false);
AdobeRGB_ AdobeRGBL = AdobeRGB_(true);
WideGamutRGB_ WideGamutRGB = WideGamutRGB_(false);
WideGamutRGB_ WideGamutRGBL = WideGamutRGB_(true);
ProPhotoRGB_ ProPhotoRGB = ProPhotoRGB_(false);
ProPhotoRGB_ ProPhotoRGBL = ProPhotoRGB_(true);
DCI_P3_RGB_ DCI_P3_RGB = DCI_P3_RGB_(false);
DCI_P3_RGB_ DCI_P3_RGBL = DCI_P3_RGB_(true);
AppleRGB_ AppleRGB = AppleRGB_(false);
AppleRGB_ AppleRGBL = AppleRGB_(true);
REC_709_RGB_ REC_709_RGB = REC_709_RGB_(false);
REC_709_RGB_ REC_709_RGBL = REC_709_RGB_(true);
REC_2020_RGB_ REC_2020_RGB = REC_2020_RGB_(false);
REC_2020_RGB_ REC_2020_RGBL = REC_2020_RGB_(true);

ColorSpaceInitial::ColorSpaceInitial()
{
    sRGB.bind(sRGBL);
    AdobeRGB.bind(AdobeRGBL);
    WideGamutRGB.bind(WideGamutRGBL);
    ProPhotoRGB.bind(ProPhotoRGBL);
    DCI_P3_RGB.bind(DCI_P3_RGBL);
    AppleRGB.bind(AppleRGBL);
    REC_709_RGB.bind(REC_709_RGBL);
    REC_2020_RGB.bind(REC_2020_RGBL);
}

Operations XYZ::cam(IO dio, CAM method)
{
    return (io == dio) ? Operations() : Operations({ Operation(cam_(io, dio, method).t()) });
}

cv::Mat XYZ::cam_(IO sio, IO dio, CAM method) const
{
    if (sio == dio)
    {
        return cv::Mat::eye(cv::Size(3, 3), CV_64FC1);
    }
    if (cams.count(std::make_tuple(dio, sio, method)) == 1)
    {
        return cams[std::make_tuple(dio, sio, method)];
    }

    // Function from http ://www.brucelindbloom.com/index.html?ColorCheckerRGB.html.
    cv::Mat XYZws = cv::Mat(illuminants.find(dio)->second);
    cv::Mat XYZWd = cv::Mat(illuminants.find(sio)->second);
    cv::Mat MA = MAs.at(method)[0];
    cv::Mat MA_inv = MAs.at(method)[1];
    cv::Mat M = MA_inv * cv::Mat::diag((MA * XYZws) / (MA * XYZWd)) * MA;
    cams[std::make_tuple(dio, sio, method)] = M;
    cams[std::make_tuple(sio, dio, method)] = M.inv();
    return M;
}

Lab::Lab(IO io_) : ColorSpace(io_, "XYZ", true)
{
    to = { Operation([this](cv::Mat src)->cv::Mat {return tosrc(src); }) };
    from = { Operation([this](cv::Mat src)->cv::Mat {return fromsrc(src); }) };
}

cv::Vec3d Lab::fromxyz(cv::Vec3d& xyz)
{
    double x = xyz[0] / illuminants.find(io)->second[0], y = xyz[1] / illuminants.find(io)->second[1], z = xyz[2] / illuminants.find(io)->second[2];
    auto f = [this](double t)->double { return t > t0 ? std::cbrtl(t) : (m * t + c); };
    double fx = f(x), fy = f(y), fz = f(z);
    return { 116. * fy - 16. ,500 * (fx - fy),200 * (fy - fz) };
}

cv::Mat Lab::fromsrc(cv::Mat& src)
{
    return channelWise(src, [this](cv::Vec3d a)->cv::Vec3d {return fromxyz(a); });
}

cv::Vec3d Lab::tolab(cv::Vec3d& lab)
{
    auto f_inv = [this](double t)->double {return t > delta ? pow(t, 3.0) : (t - c) / m; };
    double L = (lab[0] + 16.) / 116., a = lab[1] / 500., b = lab[2] / 200.;
    return { illuminants.find(io)->second[0] * f_inv(L + a),illuminants.find(io)->second[1] * f_inv(L),illuminants.find(io)->second[2] * f_inv(L - b) };
}

cv::Mat Lab::tosrc(cv::Mat& src)
{
    return channelWise(src, [this](cv::Vec3d a)->cv::Vec3d {return tolab(a); });
}

} // namespace ccm
} // namespace cv
