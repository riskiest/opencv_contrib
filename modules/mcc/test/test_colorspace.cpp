// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test
{
namespace
{

TEST(CV_ccmColorspace, test_sRGB_M)
{
    ASSERT_MAT_NEAR(sRGB.M_to, (Mat_<double>(3, 3) <<
                        0.4124564, 0.3575761, 0.1804375,
                        0.2126729, 0.7151522, 0.0721750,
                        0.0193339, 0.1191920, 0.9503041), 1e-4);
    ASSERT_MAT_NEAR(sRGB.M_from, (Mat_<double>(3, 3) <<
                        3.2404542, -1.5371385, -0.4985314,
                        -0.9692660, 1.8760108, 0.0415560,
                        0.0556434, -0.2040259, 1.0572252), 1e-4);
}

TEST(CV_ccmColorspace, test_AdobeRGB_M)
{
    ASSERT_MAT_NEAR(AdobeRGB.M_to, (Mat_<double>(3, 3) <<
                        0.5767309, 0.1855540, 0.1881852,
                        0.2973769, 0.6273491, 0.0752741,
                        0.0270343, 0.0706872, 0.9911085), 1e-4);
    ASSERT_MAT_NEAR(AdobeRGB.M_from, (Mat_<double>(3, 3) <<
                        2.0413690, -0.5649464, -0.3446944,
                        -0.9692660, 1.8760108, 0.0415560,
                        0.0134474, -0.1183897, 1.0154096), 1e-4);
}

TEST(CV_ccmColorspace, test_WideGamutRGB_M)
{
    ASSERT_MAT_NEAR(WideGamutRGB.M_to, (Mat_<double>(3, 3) <<
                        0.7161046, 0.1009296, 0.1471858,
                        0.2581874, 0.7249378, 0.0168748,
                        0.0000000, 0.0517813, 0.7734287), 1e-2);
    ASSERT_MAT_NEAR(WideGamutRGB.M_from, (Mat_<double>(3, 3) <<
                        1.4628067, -0.1840623, -0.2743606,
                        -0.5217933, 1.4472381, 0.0677227,
                        0.0349342, -0.0968930, 1.2884099), 1e-2);
}

TEST(CV_ccmColorspace, test_ProPhotoRGB_M)
{
    ASSERT_MAT_NEAR(ProPhotoRGB.M_to, (Mat_<double>(3, 3) <<
                        0.7976749, 0.1351917, 0.0313534,
                        0.2880402, 0.7118741, 0.0000857,
                        0.0000000, 0.0000000, 0.8252100), 1e-2);
    ASSERT_MAT_NEAR(ProPhotoRGB.M_from, (Mat_<double>(3, 3) <<
                        1.3459433, -0.2556075, -0.0511118,
                        -0.5445989, 1.5081673, 0.0205351,
                        0.0000000, 0.0000000, 1.2118128), 1e-2);
}

TEST(CV_ccmColorspace, test_cam_1)
{
    ASSERT_MAT_NEAR(XYZ.cam_(D50_2, D65_2), (Mat_<double>(3, 3) <<
                        0.9555766, -0.0230393, 0.0631636,
                        -0.0282895, 1.0099416, 0.0210077,
                        0.0122982, -0.0204830, 1.3299098), 1e-4);
}

TEST(CV_ccmColorspace, test_cam_2)
{
    ASSERT_MAT_NEAR(XYZ.cam_(D55_2, D50_2, VON_KRIES), (Mat_<double>(3, 3) <<
                        1.0063032, 0.0219819, -0.0223692,
                        0.0024146, 0.9981384, -0.0004869,
                        0.0000000, 0.0000000, 0.8955170), 1e-4);
}

TEST(CV_ccmColorspace, test_cam_3)
{
    ASSERT_MAT_NEAR(XYZ.cam_(D65_2, D65_2, VON_KRIES), Mat::eye(3, 3, CV_64F), 1e-4);
}

TEST(CV_ccmColorspace, test_cam_4)
{
    ASSERT_MAT_NEAR(XYZ.cam_(D65_2, D50_2, IDENTITY), (Mat_<double>(3, 3) <<
                        1.0144665, 0.0000000, 0.0000000,
                        0.0000000, 1.0000000, 0.0000000,
                        0.0000000, 0.0000000, 0.7578869), 1e-4);
}

} // namespace
} // namespace opencv_test