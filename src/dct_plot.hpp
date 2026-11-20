// =====================================================================================
// dct_plot.hpp — header file for the drawing helpers in dct_plot.cpp.
//
// WHAT A HEADER FILE IS FOR
// -------------------------
// A header only DECLARES functions: it lists their names, what they take in and what
// they give back, but not the code inside them (that lives in dct_plot.cpp).
// main.cpp does  #include "dct_plot.hpp"  so the compiler knows these functions exist
// and how to call them. The linker later joins the call in main.cpp to the actual
// code compiled from dct_plot.cpp.
//
// The three functions declared here each return a cv::Mat (an image):
//   makeHeatmapTile - coloured picture of the DCT coefficients (bottom-right tile)
//   makeTextTile    - dark tile with some text (placeholder when the DCT is off)
//   makeStatusBar   - text strip along the bottom of the window
// =====================================================================================

#pragma once
// "Only include this file once per .cpp file, even if it's #included several times."
// Without it, including the header twice (directly and via another header) could
// cause "declared twice" errors. It does the same job as the older
// #ifndef / #define / #endif "include guard" pattern.

#include <opencv2/core.hpp>   // Core OpenCV types used in the declarations below:
                              // cv::Mat (image), cv::Size (width/height), cv::String (text).
                              // Only the core module is needed here; the drawing functions
                              // (imgproc) are only needed inside dct_plot.cpp.
#include <vector>             // std::vector, for the lists of text lines

// Builds a heatmap tile of a K x K grid of values.
//   values[v*K + u] should be >= 0 (e.g. |F(u,v)|). u runs left -> right, v top -> bottom,
//   so the DC coefficient F(0,0) is the top-left cell.
//   logScale: colour ~ log(1 + value); otherwise linear, scaled to the largest
//             non-DC value (the DC cell is clipped to the top colour).
//   tileSize: size of the returned BGR image (e.g. the camera frame size, so tiles line up).
//   keep:     if > 0, outline the top-left keep x keep cells (the coefficients in use).
//
// Parameter by parameter:
//   const float* values      - pointer to the first of K*K floats. "const" = the function
//                              only reads them, it never changes them.
//   int K                    - the grid is K cells wide and K cells tall.
//   bool logScale            - true = log colouring, false = linear colouring.
//   cv::Size tileSize        - width and height the finished image must have.
//   const cv::String& title  - text for the top of the tile. Passed as a const reference
//                              (&) so the string isn't copied, and can't be changed.
//   int keep = 0             - "= 0" is a DEFAULT ARGUMENT: callers can leave it out and
//                              it will be 0 (no box drawn). main.cpp always passes it.
//                              Default values go in the header only, not in the .cpp.
// Returns: a new colour (BGR, 3-channel) image of size tileSize.
cv::Mat makeHeatmapTile(const float* values, int K, bool logScale,
                        cv::Size tileSize, const cv::String& title, int keep = 0);

// Plain dark tile with lines of text (e.g. "DCT off" placeholders).
//   const std::vector<cv::String>& lines - the text lines, top to bottom. Passed by const
//                                          reference so the list isn't copied. In main.cpp
//                                          it's written as {"DCT off", "press d to turn it on"},
//                                          and C++ builds the vector from that brace list.
//   cv::Size tileSize                    - size of the image to return.
// Returns: a new colour image of size tileSize with the text drawn on it.
cv::Mat makeTextTile(const std::vector<cv::String>& lines, cv::Size tileSize);

// Full-width strip with lines of text, for stats and key help under the tile grid.
//   width:     strip width in pixels (the width of the tile grid)
//   refHeight: height of one tile, so the text matches the tiles' text size
//
//   lines - the text lines (main.cpp passes three: stats, IDCT info, key help).
// Returns: a new colour image `width` pixels wide. Its height isn't chosen by the caller;
//          the function makes it just tall enough to fit the number of lines given.
cv::Mat makeStatusBar(const std::vector<cv::String>& lines, int width, int refHeight);
