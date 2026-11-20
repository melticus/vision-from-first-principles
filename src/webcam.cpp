// =====================================================================================
// Webcam / stream capture with OpenCV, processed on the GPU with CUDA.
// Usage:  ./webcam 0                      (local camera index)
//         ./webcam tcp://HOST:5000        (stream, e.g. from Windows into WSL)
//         ./webcam                        (uses WEBCAM_SOURCE env variable, else camera 0)
// Keys:   q / ESC quit, s save snapshot, d DCT on/off, l log/linear, +/- (or slider) coefficients,
//         1 Sobel edges, 2 Laplacian edges
//
// WHAT THIS PROGRAM DOES (big picture)
// ------------------------------------
// 1. Opens a video source (a webcam, or a network stream).
// 2. For every frame it grabs, it hands the pixels to a GPU class (GpuProcessor) that:
//      - converts the colour frame to grayscale,
//      - runs an edge-detection filter: Sobel or Laplacian (switch with keys 1 and 2),
//      - optionally computes a 2D DCT (Discrete Cosine Transform) of the image, and
//      - optionally rebuilds the image (IDCT, inverse DCT) using only the lowest N x N
//        DCT coefficients, so you can see how much of the picture survives when you
//        throw most of the frequency information away (this is the idea behind JPEG).
// 3. Shows everything in one window as a 2 x 2 grid plus a status bar:
//         [ camera frame ] [ Sobel edges  ]
//         [ IDCT rebuild ] [ DCT heatmap  ]
//         [ status bar: FPS, timings, key help ]
// 4. Lets you change N with a slider or the +/- keys, toggle the DCT, switch the
//    heatmap between log and linear scale, and save snapshots.
// =====================================================================================

// ---- Includes -----------------------------------------------------------------------
#include <opencv2/opencv.hpp>   // All of OpenCV: cv::Mat, VideoCapture, imshow, cvtColor, ...
#include <iostream>             // std::cout / std::cerr for printing messages to the terminal
#include <memory>               // std::unique_ptr / std::make_unique (smart pointer for the GPU object)
#include <string>               // std::string
#include <cstdio>               // C-style I/O (not strictly used below, harmless to include)
#include <cstdlib>              // std::getenv (reading environment variables)
#include <cmath>                // std::fabs (absolute value of a float)
#include <vector>               // std::vector (resizable arrays for the DCT coefficients)
#include <algorithm>            // std::min, std::max, std::clamp

#include "gpu_processor.cuh"    // Your own CUDA class: GpuProcessor (grayscale, Sobel, DCT, IDCT on the GPU).
                                // This header also pulls in the CUDA runtime, which is why
                                // cudaDeviceProp / cudaGetDeviceProperties work further down.
#include "dct_plot.hpp"         // Your own helper functions for drawing: makeHeatmapTile,
                                // makeTextTile and makeStatusBar.

int main(int argc, char** argv)
// argc = number of command-line words (including the program name itself)
// argv = the words themselves, e.g. argv[0] = "./webcam", argv[1] = "0"
{
    // =================================================================================
    // 1. WORK OUT WHICH VIDEO SOURCE TO OPEN
    // =================================================================================

    // Source priority: command-line argument, then WEBCAM_SOURCE env variable, then camera 0
    std::string source = "0";                 // Default: camera number 0 (the first webcam)
    if (argc > 1) {                           // If the user typed something after the program name...
        source = argv[1];                     // ...use that as the source (e.g. "1" or "tcp://...")
    } else if (const char* env = std::getenv("WEBCAM_SOURCE"); env && *env) {
        // C++17 "if with initializer": first run getenv, then test the result.
        // getenv returns nullptr if the variable doesn't exist. "env && *env" means
        // "the variable exists AND it isn't an empty string".
        source = env;                         // Use the environment variable as the source
    }
    std::cout << "Source: " << source << "\n";   // Tell the user what we're about to open

    // Decide whether the source is a camera index (only digits, e.g. "0", "12")
    // or a URL/file path (anything else, e.g. "tcp://192.168.1.5:5000").
    // find_first_not_of("0123456789") returns npos ("not found") if EVERY character is a digit.
    bool isIndex = !source.empty() &&
                   source.find_first_not_of("0123456789") == std::string::npos;

    // =================================================================================
    // 2. OPEN THE VIDEO SOURCE
    // =================================================================================

    cv::VideoCapture cap;                     // OpenCV object that reads frames from a camera/stream/file
    if (isIndex) {
        int camIndex = std::stoi(source);     // Convert the text "0" into the integer 0
#ifdef _WIN32
        // Compiled on Windows: use the DirectShow backend (usually the most reliable there)
        cap.open(camIndex, cv::CAP_DSHOW);
#else
        // Compiled on Linux: use Video4Linux2, the standard Linux camera interface
        cap.open(camIndex, cv::CAP_V4L2);
#endif
        // Ask the camera for 1280 x 720 (720p). This is a request, not a guarantee:
        // if the camera can't do it, it picks the closest size it supports.
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    } else {
        // Not a number, so treat it as a URL or file and decode it with FFmpeg
        // (handles tcp://, rtsp://, http://, .mp4 files, etc.)
        cap.open(source, cv::CAP_FFMPEG);
    }

    if (!cap.isOpened()) {                    // Opening failed (wrong index, no camera, bad URL...)
        std::cerr << "Error: could not open source " << source << "\n";
        return 1;                             // Non-zero return code = the program failed
    }

    // =================================================================================
    // 3. CHECK THAT A CUDA GPU IS AVAILABLE
    // =================================================================================

    // Report which GPU we're running on
    cudaDeviceProp prop{};                    // Struct the CUDA runtime fills with GPU info; {} zeroes it
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        // Ask for the properties of GPU number 0. If this fails there's no usable
        // NVIDIA GPU / driver, so nothing else in the program can work.
        std::cerr << "Error: no CUDA device found (check nvidia-smi)\n";
        return 1;
    }
    std::cout << "GPU: " << prop.name << "\n"   // prop.name is the GPU's model name, e.g. "NVIDIA GeForce RTX 3060"
              << "Keys: q/ESC quit | s snapshot | d DCT on/off | l log/linear | +/- coefficients"
              << " | 1 Sobel | 2 Laplacian\n";

    // =================================================================================
    // 4. SET UP IMAGES, BUFFERS AND SETTINGS USED IN THE LOOP
    // =================================================================================

    // cv::Mat is OpenCV's image type. Declaring them once here (outside the loop) lets
    // OpenCV reuse their memory every frame instead of allocating new memory each time.
    //   frame      - the colour (BGR) image straight from the camera
    //   edges      - the 1-channel grayscale Sobel edge image from the GPU
    //   edgesBgr   - edges converted to 3 channels so it can sit next to the colour frame
    //   recon      - 1-channel image rebuilt by the IDCT on the GPU
    //   reconTile  - recon converted to 3 channels for display
    //   heatTile   - the coloured picture of the DCT coefficients (heatmap)
    //   statusBar  - the text strip at the bottom of the window
    //   topRow     - frame + edgesBgr side by side
    //   bottomRow  - reconTile + heatTile side by side
    //   view       - the final full window image (top row, bottom row, status bar stacked)
    cv::Mat frame, edges, edgesBgr, recon, reconTile, heatTile, statusBar, topRow, bottomRow, view;

    // Smart pointer to the GPU processor. It starts empty (nullptr) because we don't
    // know the frame size yet; it is created inside the loop once the first frame arrives.
    // unique_ptr automatically frees the GPU object (and its GPU memory) when replaced
    // or when the program ends.
    std::unique_ptr<GpuProcessor> gpu;

    // K is the size of the block of DCT coefficients the GPU sends back (K x K values).
    // constexpr = known at compile time. DCT_K is defined inside GpuProcessor.
    constexpr int K = GpuProcessor::DCT_K;

    std::vector<float> coef(K * K);          // raw F(u,v) from the GPU, coef[v*K + u]
                                             // (a 2D K x K grid stored as one flat 1D array,
                                             //  row v, column u -> index v*K + u)
    std::vector<float> smoothed(K * K, 0);   // |F(u,v)| smoothed over time, for a steadier heatmap
                                             // (starts as all zeros)
    // Which edge filter to use. Kept here in main (not only inside the GpuProcessor) because
    // the GpuProcessor is thrown away and re-created if the frame size changes, and we
    // don't want the choice to reset to Sobel when that happens.
    EdgeFilter edgeFilter = EdgeFilter::Sobel;
    bool showDct = true, logScale = true;    // Start with the DCT turned on and the heatmap in log scale.
                                             // Log scale is needed because the DC coefficient is
                                             // MUCH bigger than the rest; on a linear scale
                                             // everything else would look black.

    // ---- Create the window and the slider -------------------------------------------

    // Window with a slider choosing how many coefficients (N, for N x N) the IDCT uses
    const std::string win = "Webcam | CUDA";            // Window title (also used as its ID)
    const std::string slider = "Coefficients N";        // Slider label (also used as its ID)
    int keep = std::min(16, K);                         // N = how many coefficients to keep per side.
                                                        // Start at 16, unless K is smaller than 16.
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);          // Create the window; AUTOSIZE = window fits the image
    cv::createTrackbar(slider, win, nullptr, K);        // Add a slider going 0..K. nullptr = we don't link it
                                                        // to a variable; we read it ourselves each frame instead.
    cv::setTrackbarMin(slider, win, 1);                 // Don't allow 0 (keeping 0 coefficients makes no sense)
    cv::setTrackbarPos(slider, win, keep);              // Put the slider at the starting value (16)

    // ---- Counters for snapshots and FPS ---------------------------------------------

    int snapCount = 0;                                  // Used to number snapshot files: snapshot_0.png, _1, ...
    double tickFreq = cv::getTickFrequency();           // How many "ticks" the system clock does per second
    int64 lastTick = cv::getTickCount();                // Clock reading now; used to time each loop iteration

    // =================================================================================
    // 5. MAIN LOOP — runs once per video frame until the user quits
    // =================================================================================
    while (true) {

        // ---- 5a. Grab a frame -------------------------------------------------------
        if (!cap.read(frame) || frame.empty()) {
            // read() grabs and decodes the next frame into 'frame'. It returns false if
            // the camera disconnected or the stream ended.
            std::cerr << "Error: empty frame grabbed\n";
            break;                                      // Leave the loop and clean up
        }
        if (!frame.isContinuous()) frame = frame.clone();   // GPU copy needs one contiguous block
        // A cv::Mat can have padding at the end of each row. The GPU copy treats the
        // image as one unbroken block of bytes, so if there's padding, clone() makes
        // a tightly packed copy.

        // ---- 5b. Create (or re-create) the GPU processor if needed ------------------
        // Allocate GPU buffers once we know the frame size (re-allocate if it changes)
        if (!gpu || edges.cols != frame.cols || edges.rows != frame.rows) {
            // True on the very first frame (gpu is still empty), or if the stream
            // suddenly changes resolution (edges was sized to the OLD frame).
            gpu = std::make_unique<GpuProcessor>(frame.cols, frame.rows);  // Allocates GPU memory for this size
            edges.create(frame.rows, frame.cols, CV_8UC1);  // CPU image for edges: 8-bit, 1 channel (grayscale)
            recon.create(frame.rows, frame.cols, CV_8UC1);  // CPU image for the IDCT result, same format
            std::cout << "Frame size: " << frame.cols << "x" << frame.rows << "\n";
        }

        // ---- 5c. Read the slider ----------------------------------------------------
        // Read the slider (it may have been dragged since the last frame)
        keep = std::clamp(cv::getTrackbarPos(slider, win), 1, K);
        // clamp forces the value into the range [1, K], just in case.

        // ---- 5d. Do all the heavy maths on the GPU ----------------------------------
        // Tell the GPU processor which edge filter to run this frame (Sobel or Laplacian)
        gpu->setEdgeFilter(edgeFilter);

        // One GPU pass: grayscale + edge filter, plus DCT and IDCT when switched on
        gpu->process(frame.ptr<uint8_t>(),                          // input: pointer to the camera pixels
                     edges.ptr<uint8_t>(),                          // output: where to write the Sobel edges
                     showDct ? coef.data() : nullptr,               // output: where to write the K x K DCT
                                                                    //   coefficients (nullptr = skip the DCT)
                     showDct ? recon.ptr<uint8_t>() : nullptr,      // output: where to write the IDCT image
                                                                    //   (nullptr = skip the IDCT)
                     keep);                                         // how many coefficients (N x N) the IDCT keeps
        // "condition ? a : b" means "if condition then a else b". Passing nullptr is
        // how we tell the GPU code not to bother with the DCT/IDCT when it's off.

        // ---- 5e. Build the two bottom tiles -----------------------------------------
        if (showDct) {
            // Turn the 1-channel IDCT image into 3 channels so it can be joined to colour images
            cv::cvtColor(recon, reconTile, cv::COLOR_GRAY2BGR);

            // Exponential moving average of |F|: 70% previous + 30% new, to reduce flicker
            // Each coefficient's displayed value moves only 30% of the way towards the
            // new value each frame, so camera noise doesn't make the heatmap jump around.
            for (int i = 0; i < K * K; ++i)
                smoothed[i] = 0.7f * smoothed[i] + 0.3f * std::fabs(coef[i]);
                // fabs because DCT coefficients can be negative; we only plot their size

            // Build the heatmap title text, e.g. "DCT |F(u,v)|, lowest 32x32 (log), box = kept"
            // cv::format works like printf but returns a std::string.
            std::string title = cv::format("DCT |F(u,v)|, lowest %dx%d (%s), box = kept",
                                           K, K, logScale ? "log" : "linear");

            // Draw the coloured heatmap of the smoothed coefficients, the same size as the
            // camera frame, with a box showing which N x N coefficients are being kept.
            heatTile = makeHeatmapTile(smoothed.data(), K, logScale, frame.size(), title, keep);
        } else {
            // DCT is off: show placeholder tiles with a message instead
            reconTile = makeTextTile({"IDCT off", "press d to turn it on"}, frame.size());
            heatTile  = makeTextTile({"DCT off",  "press d to turn it on"}, frame.size());
        }

        // ---- 5f. Work out the numbers for the status bar ----------------------------
        // Stats and key help go in a status bar, so nothing is drawn over the images
        int64 now = cv::getTickCount();                                // Clock reading now
        double fps = tickFreq / static_cast<double>(now - lastTick);   // ticks-per-second / ticks-this-frame
                                                                       // = frames per second (for the whole loop,
                                                                       // including camera, GPU and drawing)
        lastTick = now;                                                // Remember for next frame

        // cv::format returns a std::string sized to fit, so nothing can be cut short
        // Name of the edge filter currently in use, shown in the status bar
        const char* filterName = (edgeFilter == EdgeFilter::Laplacian) ? "Laplacian" : "Sobel";
        std::string stats = cv::format(
            "FPS %.0f  |  %s %.2f ms  |  DCT %.2f ms  |  IDCT %.2f ms  |  DC %.0f  |  %s",
            fps,                      // %.0f  -> FPS with no decimals
            filterName,               // %s    -> "Sobel" or "Laplacian"
            gpu->lastGpuMs(),         // %.2f  -> how long the grayscale+edge-filter GPU step took, in ms
            gpu->lastDctMs(),         // %.2f  -> how long the DCT took on the GPU, in ms
            gpu->lastIdctMs(),        // %.2f  -> how long the IDCT took on the GPU, in ms
            coef[0],                  // %.0f  -> F(0,0), the DC coefficient: proportional to the image's
                                      //          average brightness (keeps its last value when DCT is off)
            prop.name);               // %s    -> GPU name

        // What percentage of ALL possible DCT coefficients are we keeping?
        // A full-image DCT has one coefficient per pixel (cols x rows), and we keep N x N.
        // static_cast<double> avoids integer overflow / integer division.
        double keptPct = 100.0 * keep * keep / (static_cast<double>(frame.cols) * frame.rows);
        std::string recInfo = cv::format(
            "Bottom left: image rebuilt from the lowest %dx%d = %d DCT coefficients (%.3f%% of all)",
            keep, keep, keep * keep, keptPct);     // %% prints a literal % sign
        std::string help =
            "Keys:  q quit   s snapshot   d DCT on/off   l log/linear   + / - (or slider) change N"
            "   1 Sobel   2 Laplacian";
            // Two string literals next to each other are joined into one by the compiler.

        // ---- 5g. Assemble the final window image ------------------------------------
        // Layout:   camera       | Sobel
        //           IDCT image   | DCT heatmap
        //           status bar (full width)
        cv::cvtColor(edges, edgesBgr, cv::COLOR_GRAY2BGR);   // edges: 1 channel -> 3 channels
                                                             // (hconcat needs matching channel counts)
        cv::hconcat(frame, edgesBgr, topRow);                // hconcat = join side by side (horizontal)
        cv::hconcat(reconTile, heatTile, bottomRow);
        // Build the status bar as wide as the whole top row, with three lines of text
        statusBar = makeStatusBar({stats, recInfo, help}, topRow.cols, frame.rows);
        // vconcat = stack on top of each other (vertical). All three must be the same width.
        cv::vconcat(std::vector<cv::Mat>{topRow, bottomRow, statusBar}, view);

        // ---- 5h. Shrink to fit the screen and show ----------------------------------
        // Shrink to fit a typical screen (max 1600 x 900), keeping the aspect ratio
        // Take the smallest of: 1.0 (never enlarge), the factor that makes it 1600 wide,
        // and the factor that makes it 900 tall. Using the same factor for both
        // directions keeps the picture from being stretched.
        double scale = std::min({1.0, 1600.0 / view.cols, 900.0 / view.rows});
        if (scale < 1.0) cv::resize(view, view, {}, scale, scale, cv::INTER_AREA);
        // {} = no target size given, so resize uses the scale factors instead.
        // INTER_AREA is the best-looking interpolation for shrinking images.
        cv::imshow(win, view);                               // Put the image in the window

        // ---- 5i. Handle keyboard input ----------------------------------------------
        int key = cv::waitKey(1) & 0xFF;
        // waitKey(1) waits up to 1 ms for a key press AND lets OpenCV actually redraw the
        // window (imshow shows nothing without it). It returns -1 if no key was pressed.
        // "& 0xFF" keeps only the lowest 8 bits, so the key code is the plain ASCII value
        // on every platform.
        if (key == 'q' || key == 27) break;                  // q or ESC (ASCII 27) -> quit the loop
        if (key == 'd') showDct = !showDct;                  // d -> toggle DCT/IDCT on/off
        if (key == 'l') logScale = !logScale;                // l -> toggle heatmap log/linear scale
        if (key == '1') edgeFilter = EdgeFilter::Sobel;      // 1 -> Sobel edges (top-right tile)
        if (key == '2') edgeFilter = EdgeFilter::Laplacian;  // 2 -> Laplacian edges (top-right tile)
        // '1' and '2' are the characters (ASCII 49 and 50), not the numbers 1 and 2.
        // Number-pad keys may send different codes on some systems; use the keys above the letters.
        // + (or = , the same key without Shift) -> one more coefficient, but not above K.
        // We move the slider rather than 'keep' directly, so the slider stays in sync
        // and the new value is picked up at the top of the next frame.
        if (key == '+' || key == '=') cv::setTrackbarPos(slider, win, std::min(keep + 1, K));
        // - (or _ , the same key with Shift) -> one fewer coefficient, but not below 1
        if (key == '-' || key == '_') cv::setTrackbarPos(slider, win, std::max(keep - 1, 1));
        if (key == 's') {                                    // s -> save the whole window as a PNG
            std::string name = "snapshot_" + std::to_string(snapCount++) + ".png";
            // snapCount++ uses the current number, then adds 1 for next time
            cv::imwrite(name, view);                         // Saved in the folder the program was run from
            std::cout << "Saved " << name << "\n";
        }
    }   // end of while loop -> go back and grab the next frame

    // =================================================================================
    // 6. CLEAN UP
    // =================================================================================
    cap.release();              // Close the camera / stream
    cv::destroyAllWindows();    // Close the OpenCV window
    return 0;                   // 0 = the program finished normally
    // 'gpu' (the unique_ptr) is destroyed automatically here, freeing the GPU memory.
}
