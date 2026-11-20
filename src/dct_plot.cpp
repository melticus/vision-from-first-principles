// =====================================================================================
// dct_plot.cpp — drawing helpers for the webcam program (runs on the CPU, plain OpenCV).
//
// WHAT THIS FILE DOES (big picture)
// ---------------------------------
// main() builds its window out of "tiles" (images the same size as the camera frame).
// This file makes the tiles that aren't just camera/GPU images:
//   makeHeatmapTile - the bottom-right tile: a coloured K x K grid showing how big each
//                     DCT coefficient |F(u,v)| is, with axis numbers, a colour bar,
//                     a title and a white box around the coefficients the IDCT is using.
//   makeTextTile    - a plain dark tile with a few lines of text (used when the DCT is off).
//   makeStatusBar   - the strip along the bottom of the window with FPS, timings and key help.
//
// USEFUL OPENCV FACTS FOR READING THIS FILE
// -----------------------------------------
//   cv::Mat           - an image (or any 2D array).
//   CV_8UC3 / CV_8UC1 - 8-bit unsigned, 3 channels (colour) / 1 channel (gray).
//   cv::Scalar(B,G,R) - a colour. OpenCV uses Blue, Green, Red order, not RGB.
//   cv::Point(x, y)   - a position; x goes right, y goes DOWN (0,0 is the top-left corner).
//   cv::Size(w, h)    - a width and height.
//   cv::Rect(x,y,w,h) - a rectangle: top-left corner plus width and height.
//   cvRound(x)        - round a double to the nearest int.
// =====================================================================================

#include "dct_plot.hpp"            // Declarations of the three public functions below
#include <opencv2/imgproc.hpp>     // Drawing and image processing: putText, line, rectangle, resize, applyColorMap
#include <algorithm>               // std::min, std::max
#include <cmath>                   // std::log1p, std::expm1
#include <cstdio>                  // std::snprintf (formatting numbers into text)
#include <vector>                  // std::vector

// ---- Private helpers ----------------------------------------------------------------
// An unnamed ("anonymous") namespace: everything inside is only visible in THIS file.
// It stops these short names (text, fmt, ...) clashing with names in other files.
namespace {

const cv::Scalar kBackground(30, 30, 30);    // Very dark gray, used as the tile background
const cv::Scalar kText(230, 230, 230);       // Near-white, used for all text
const int kFont = cv::FONT_HERSHEY_SIMPLEX;  // OpenCV's basic built-in font
// (The "k" prefix is a common naming habit meaning "constant".)

// Font size and line thickness that scale with the tile, so text stays readable
// after the whole window is shrunk to fit the screen.
// Designed for a 720-pixel-tall tile: there the font scale is 0.8 and lines are 2 px thick.
// A taller tile gets proportionally bigger text, a shorter one smaller text.
double fontScale(cv::Size s) { return 0.8 * s.height / 720.0; }
int    thickness(cv::Size s) { return std::max(1, cvRound(2.0 * s.height / 720.0)); }  // never thinner than 1 px

// Shortcut for drawing text in the standard font and colour.
//   img   - the image to draw on (passed by reference, so it's changed in place)
//   s     - the text
//   org   - position of the text's BOTTOM-LEFT corner (that's how putText works)
//   scale - font size, th - stroke thickness
void text(cv::Mat& img, const cv::String& s, cv::Point org, double scale, int th)
{
    cv::putText(img, s, org, kFont, scale, kText, th, cv::LINE_AA);   // LINE_AA = anti-aliased (smooth edges)
}

// Turn a number into short text: no decimals for 100 or more ("1532"),
// one decimal below 100 ("12.4"). Used for the colour bar labels.
cv::String fmt(float v)
{
    char buf[32];                        // small text buffer, plenty for one number
    // snprintf writes formatted text into buf, never more than sizeof(buf) characters.
    // The ?: picks which format string to use depending on the size of v.
    std::snprintf(buf, sizeof(buf), v >= 100.0f ? "%.0f" : "%.1f", v);
    return buf;                          // converted automatically to a cv::String
}

} // namespace

// =====================================================================================
// makeHeatmapTile — draws the DCT coefficient heatmap
//   values    - K x K array of |F(u,v)| values (already smoothed in main), values[v*K + u]
//   K         - size of the grid
//   logScale  - true = colour by log(1 + |F|), false = colour by |F| directly
//   tileSize  - how big the finished tile must be (same as the camera frame)
//   title     - text for the top of the tile
//   keep      - N, the number of coefficients per side the IDCT uses (for the white box)
// Returns the finished colour image.
// =====================================================================================
cv::Mat makeHeatmapTile(const float* values, int K, bool logScale,
                        cv::Size tileSize, const cv::String& title, int keep)
{
    cv::Mat tile(tileSize, CV_8UC3, kBackground);   // Blank dark colour image to draw on
    const double fs = fontScale(tileSize);          // font size for this tile
    const int th = thickness(tileSize);             // line thickness for this tile
    const int lineH = cvRound(40 * fs);             // height of one line of text, in pixels

    // ---- 1. Normalise values to [0, 1] ---------------------------------------
    // Colours are picked from a 0..1 (later 0..255) range, so every value is divided
    // by the biggest one.
    std::vector<float> n(K * K);                    // the normalised copy
    float maxV = 1e-6f;                             // start with a tiny positive number, never 0,
                                                    // so the division below can't divide by zero
    for (int i = 0; i < K * K; ++i) {
        // log1p(x) = ln(1 + x). Squashes big numbers together so small and large
        // coefficients can both be seen, and gives 0 for x = 0 (plain log(0) would be -infinity).
        n[i] = logScale ? std::log1p(values[i]) : values[i];
        if (logScale || i != 0) maxV = std::max(maxV, n[i]);   // linear: DC would wash out the rest
        // i == 0 is the DC coefficient F(0,0), which is far larger than all the others.
        // On a linear scale, if it set the maximum, every other cell would be nearly black.
        // So in linear mode it's left out when finding the max (and gets clipped to 1 below).
    }
    for (float& x : n) x = std::min(x / maxV, 1.0f);   // divide by max; anything over 1 (DC in linear mode) -> 1
    // "float& x" means x refers to the actual element, so changing x changes n.

    // ---- 2. K x K image of colours -------------------------------------------
    cv::Mat small(K, K, CV_8UC1);                   // tiny K x K grayscale image, one pixel per coefficient
    for (int v = 0; v < K; ++v)                     // row = vertical frequency v
        for (int u = 0; u < K; ++u)                 // column = horizontal frequency u
            // Scale 0..1 up to 0..255. saturate_cast rounds and clamps safely into a byte.
            small.at<uchar>(v, u) = cv::saturate_cast<uchar>(n[v * K + u] * 255.0f);
    cv::Mat coloured;
    // Map each gray level to a colour: Viridis goes dark purple (0) -> blue -> green -> yellow (255).
    // Output is a 3-channel colour image.
    cv::applyColorMap(small, coloured, cv::COLORMAP_VIRIDIS);

    // ---- 3. Layout: square map, colour bar to its right ----------------------
    // Work out how much room each part needs, all scaled with the font size.
    const int top = 3 * lineH;                    // title + u-axis label + tick numbers
    const int bottom = lineH;                     // room under the map
    const int barW = cvRound(25 * fs),            // width of the colour bar
              barGap = cvRound(25 * fs),          // gap between the map and the colour bar
              barLabels = cvRound(110 * fs);      // room for the numbers to the right of the bar
    const int leftLabels = cvRound(60 * fs);      // room for the v tick numbers left of the map

    // cell = size in pixels of one coefficient's square. Pick the biggest whole number
    // that fits BOTH the available height and the available width (so the map stays square).
    int cell = std::min((tileSize.height - top - bottom) / K,
                        (tileSize.width - leftLabels - barGap - barW - barLabels) / K);
    if (cell < 1) return tile;                    // tile too small to draw into
    const int S = cell * K;                       // side length of the whole square map in pixels
    // x position of the map: centre the whole group [labels | map | gap | bar | bar labels]
    // in the tile, then step past the left labels.
    const int mapX = (tileSize.width - (leftLabels + S + barGap + barW + barLabels)) / 2 + leftLabels;
    const int mapY = top;                         // map starts just below the title area

    // Blow each value up to a cell x cell square (nearest neighbour keeps hard edges)
    // (Smooth interpolation would blur neighbouring coefficients into each other.)
    cv::Mat big;
    cv::resize(coloured, big, cv::Size(S, S), 0, 0, cv::INTER_NEAREST);
    // tile(cv::Rect(...)) is a "view" of just that rectangle inside tile (no copy is made).
    // copyTo writes 'big' into that rectangle, so it lands in the right place on the tile.
    big.copyTo(tile(cv::Rect(mapX, mapY, S, S)));

    // Thin grid lines between cells (skipped when cells are small, where they'd just add clutter)
    if (cell >= 12) {
        for (int i = 1; i < K; ++i) {             // K-1 inner lines in each direction
            // Vertical line at the left edge of column i, from top to bottom of the map
            cv::line(tile, {mapX + i * cell, mapY}, {mapX + i * cell, mapY + S - 1}, kBackground, 1);
            // Horizontal line at the top edge of row i, from left to right of the map
            cv::line(tile, {mapX, mapY + i * cell}, {mapX + S - 1, mapY + i * cell}, kBackground, 1);
        }
    }

    // Box around the coefficients currently used by the IDCT
    // (The kept coefficients are the top-left keep x keep corner: the lowest frequencies.)
    if (keep > 0) {
        int k = std::min(keep, K);                // never draw a box bigger than the map
        cv::rectangle(tile, cv::Rect(mapX, mapY, k * cell, k * cell),
                      cv::Scalar(255, 255, 255),  // white
                      std::max(2, th),            // at least 2 px thick so it stands out
                      cv::LINE_AA);
    }

    // ---- 4. Axes: tick numbers at 0, K/2, K-1 ---------------------------------
    const double tfs = 0.75 * fs;                 // tick labels are a bit smaller than the main text
    for (int t : {0, K / 2, K - 1}) {             // loop over just these three values
        cv::String s = std::to_string(t);         // the number as text
        int cxPix = mapX + t * cell + cell / 2;   // x of the centre of column t
        int cyPix = mapY + t * cell + cell / 2;   // y of the centre of row t
        int base = 0;                             // getTextSize fills this with the baseline offset (unused)
        cv::Size ts = cv::getTextSize(s, kFont, tfs, th, &base);   // how big the text will be when drawn
        // u ticks go above the map, centred over their column
        text(tile, s, {cxPix - ts.width / 2, mapY - cvRound(8 * fs)}, tfs, th);                 // u ticks (top)
        // v ticks go left of the map, right-aligned, and vertically centred on their row
        text(tile, s, {mapX - ts.width - cvRound(10 * fs), cyPix + ts.height / 2}, tfs, th);    // v ticks (left)
    }
    // Axis names: u above the tick numbers, v underneath the map
    text(tile, "u  (horizontal frequency) ->", {mapX, mapY - cvRound(lineH * 0.9)}, tfs, th);
    text(tile, "v  (vertical frequency) runs down", {mapX, mapY + S + cvRound(lineH * 0.75)}, tfs, th);

    // ---- 5. Colour bar with value labels --------------------------------------
    // A vertical strip showing the colour scale: yellow (largest) at the top, purple (0) at the bottom.
    const int barX = mapX + S + barGap;           // x position: to the right of the map
    cv::Mat ramp(S, 1, CV_8UC1);                  // a 1-pixel-wide column, S pixels tall
    // Fill it with a smooth ramp: 255 at the top row (y = 0) down to 0 at the bottom row.
    // (Note: if S were 1 this would divide by zero, but that only happens for a 1x1 map.)
    for (int y = 0; y < S; ++y) ramp.at<uchar>(y, 0) = cv::saturate_cast<uchar>(255.0 * (S - 1 - y) / (S - 1));
    cv::Mat rampCol, rampBig;
    cv::applyColorMap(ramp, rampCol, cv::COLORMAP_VIRIDIS);   // same colours as the map
    cv::resize(rampCol, rampBig, cv::Size(barW, S), 0, 0, cv::INTER_NEAREST);   // widen to barW pixels
    rampBig.copyTo(tile(cv::Rect(barX, mapY, barW, S)));      // paste it next to the map

    // Label the bar in real |F| units (undo the log for the top label)
    // In log mode the colours were based on ln(1 + |F|), so expm1(x) = e^x - 1 converts
    // back to a real |F| value. The top of the bar is the max; the middle is half the max
    // in log terms (which is a much smaller |F| than half of topVal).
    float topVal = logScale ? std::expm1(maxV) : maxV;
    float midVal = logScale ? std::expm1(0.5f * maxV) : 0.5f * maxV;
    const int lx = barX + barW + cvRound(8 * fs);   // x position for the labels, just right of the bar
    // In linear mode the top label gets a "+", because DC (and anything bigger than the max)
    // was clipped to the top colour, so the top colour really means "this value or more".
    text(tile, fmt(topVal) + (logScale ? "" : "+"), {lx, mapY + cvRound(15 * fs)}, tfs, th);   // top label
    text(tile, fmt(midVal), {lx, mapY + S / 2 + cvRound(8 * fs)}, tfs, th);                    // middle label
    text(tile, "0", {lx, mapY + S}, tfs, th);                                                  // bottom label

    // ---- 6. Title ---------------------------------------------------------------
    text(tile, title, {cvRound(15 * fs), cvRound(lineH * 0.8)}, fs, th);   // top-left, full-size font
    return tile;
}

// =====================================================================================
// makeTextTile — a dark tile with a few lines of text on it
//   lines    - the lines to write, top to bottom (an empty string leaves a blank line)
//   tileSize - size of the tile (same as the camera frame)
// Used by main() for the "DCT off / press d to turn it on" placeholders.
// =====================================================================================
cv::Mat makeTextTile(const std::vector<cv::String>& lines, cv::Size tileSize)
{
    cv::Mat tile(tileSize, CV_8UC3, kBackground);   // blank dark tile
    const double fs = fontScale(tileSize);
    const int th = thickness(tileSize);
    const int lineH = cvRound(45 * fs);             // spacing between lines
    int y = cvRound(lineH * 1.2);                   // y of the first line's baseline (a little below the top)
    for (const auto& s : lines) {                   // for each line of text...
        if (!s.empty()) text(tile, s, {cvRound(25 * fs), y}, fs, th);   // ...draw it (skip empty ones)
        y += lineH;                                 // move down for the next line
    }
    return tile;
}

// =====================================================================================
// makeStatusBar — the text strip along the bottom of the window
//   lines     - the lines to show (main passes: stats, IDCT info, key help)
//   width     - how wide the bar must be (the full width of the window)
//   refHeight - a tile height (main passes frame.rows). Used only to size the text, so
//               the bar's text matches the text in the tiles above it.
// The bar's own height is calculated to fit exactly the number of lines given.
// =====================================================================================
cv::Mat makeStatusBar(const std::vector<cv::String>& lines, int width, int refHeight)
{
    const cv::Size ref(width, refHeight);         // size the text like the tiles above
    const double fs = fontScale(ref);
    const int th = thickness(ref);
    const int lineH = cvRound(45 * fs);           // spacing between lines
    // One line height per line of text, plus a bit of padding at the bottom.
    // static_cast<int> because lines.size() is an unsigned size_t.
    const int height = lineH * static_cast<int>(lines.size()) + cvRound(0.6 * lineH);

    // Background slightly lighter (45) than the tiles (30) so the bar stands out.
    cv::Mat bar(height, width, CV_8UC3, cv::Scalar(45, 45, 45));
    int y = lineH;                                // baseline of the first line
    for (const auto& s : lines) {
        text(bar, s, {cvRound(25 * fs), y}, fs, th);   // draw the line, indented a little from the left
        y += lineH;                                    // next line goes one line height lower
    }
    return bar;
}
