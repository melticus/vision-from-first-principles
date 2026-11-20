// =====================================================================================
// gpu_processor.cuh — header file for the GpuProcessor class (code is in gpu_processor.cu).
//
// WHAT THIS FILE DOES
// -------------------
// It declares the GpuProcessor class: what it stores (pointers to GPU memory, a CUDA
// stream, timing events) and what you can do with it (create it, run process() on a
// frame, read the timings). main.cpp includes this so it can create and use a GpuProcessor.
//
// ".cuh" is just the usual name ending for a CUDA header. It's included both by the
// CUDA file (gpu_processor.cu, compiled by nvcc) and by main.cpp, so everything in it
// must be plain C++ that any compiler understands: no __global__ kernels in here.
//
// NAMING HABITS USED BELOW
//   name_    - a trailing underscore marks a private member variable of the class.
//   d_name_  - the "d_" means the pointer points to DEVICE (GPU) memory. The CPU must
//              never read or write through these pointers directly; it can only pass
//              them to CUDA functions and kernels.
// =====================================================================================

#pragma once                // Only include this file once per .cpp/.cu file (avoids duplicate definitions)
#include <cstdint>          // Fixed-size integer types: uint8_t = unsigned 8-bit (a 0..255 byte, one pixel channel)
#include <cuda_runtime.h>   // The CUDA runtime API: cudaStream_t, cudaEvent_t, cudaMalloc, etc.
                            // Including it here is also why main.cpp can use cudaDeviceProp
                            // and cudaGetDeviceProperties without including it itself.

// Which edge filter process() runs on the grayscale image.
// "enum class" makes a small set of named options. Unlike a plain enum, the names must be
// written with the type in front (EdgeFilter::Sobel), so they can't clash with other names.
enum class EdgeFilter {
    Sobel,       // first derivative: gradient magnitude, thick smooth edges
    Laplacian    // second derivative: thin edges, but more sensitive to noise
};

// Owns the GPU buffers for one frame size and runs the CUDA pipeline on each frame:
//   colour -> grayscale -> Sobel edges
//                       -> DCT (lowest DCT_K x DCT_K coefficients of the whole image)
//                          -> IDCT (image rebuilt from only those coefficients)
// "Owns" means: the constructor allocates the GPU memory, the destructor frees it, and
// nothing else should. If the frame size changes, main.cpp throws this object away and
// makes a new one, because every buffer is sized for one particular width x height.
class GpuProcessor {
public:   // ---- Everything below here can be used from outside the class (e.g. by main.cpp)

    // Number of DCT frequencies computed in each direction: the DCT always computes
    // DCT_K x DCT_K coefficients, and the IDCT can use any lowest N x N of them (N <= DCT_K).
    // Work grows with DCT_K, and the image must be at least DCT_K pixels in each direction.
    //   static    - belongs to the class itself, not to each object, so it can be read as
    //               GpuProcessor::DCT_K without having an object (main.cpp does this).
    //   constexpr - a compile-time constant, so it can size arrays and be used in GPU kernels.
    // At 256, the heatmap shows a 256 x 256 grid and the slider in main.cpp goes up to 256.
    static constexpr int DCT_K = 256;

    // Constructor: called when the object is created, e.g. std::make_unique<GpuProcessor>(1280, 720).
    // Allocates all GPU memory for this frame size and builds the cosine tables.
    GpuProcessor(int width, int height);
    // Destructor (the ~ means destructor): called automatically when the object is
    // destroyed. Frees all the GPU memory, the stream and the events.
    ~GpuProcessor();

    // Forbid copying. "= delete" means "this function does not exist; using it is a
    // compile error". Why: a copy would hold the SAME GPU pointers as the original.
    // When both were destroyed, each would cudaFree the same memory -> crash/corruption.
    GpuProcessor(const GpuProcessor&) = delete;              // no copy constructor:   GpuProcessor b = a;
    GpuProcessor& operator=(const GpuProcessor&) = delete;   // no copy assignment:    b = a;
    // (That's why main.cpp holds it in a std::unique_ptr: the pointer can be replaced
    //  without ever copying the object itself.)

    // Runs the whole GPU pipeline on one frame. Called once per frame from main.cpp.
    // "host pointer" = a pointer to normal CPU memory (here, the data inside a cv::Mat).
    // bgr:   host pointer, width*height*3 bytes (OpenCV's default BGR layout)
    // edges: host pointer, width*height bytes (output, single channel)
    // dct:   optional host pointer, DCT_K*DCT_K floats (output, dct[v*DCT_K + u] = F(u,v))
    // recon: optional host pointer, width*height bytes (output): the IDCT of the lowest
    //        keep x keep coefficients, i.e. a low-pass version of the grayscale image
    // keep:  how many coefficients per direction the IDCT uses (clamped to 1..DCT_K)
    // Pass nullptr for dct and/or recon to skip them (the DCT still runs if recon is wanted).
    //
    // The "= nullptr" and "= DCT_K" are default arguments: process(bgr, edges) on its own
    // does only grayscale + Sobel; adding dct/recon/keep turns on the rest.
    // bgr is "const" because process() only reads the camera frame; the others are outputs.
    void process(const uint8_t* bgr, uint8_t* edges,
                 float* dct = nullptr, uint8_t* recon = nullptr, int keep = DCT_K);

    // Choose which edge filter process() uses from the next frame on (default: Sobel).
    // main.cpp calls this when 1 or 2 is pressed.
    void setEdgeFilter(EdgeFilter f) { edgeFilter_ = f; }
    // Read back the current choice (used by main.cpp to label the status bar).
    EdgeFilter edgeFilter() const { return edgeFilter_; }

    // GPU time for the last frame, in ms:
    //   lastGpuMs:  upload + grayscale + edge filter (Sobel or Laplacian) + edge download
    //   lastDctMs:  the two DCT passes + coefficient download (0 if skipped)
    //   lastIdctMs: the two IDCT passes + image download (0 if skipped)
    // ("0 if skipped" in practice means very close to 0, since the two timing markers
    //  are recorded back to back.)
    // These are "getters": short functions written right here in the header that just
    // return a private value. The "const" after () promises they don't change the object.
    float lastGpuMs()  const { return lastMs_; }
    float lastDctMs()  const { return lastDctMs_; }
    float lastIdctMs() const { return lastIdctMs_; }

private:  // ---- Everything below here can only be used inside the class's own functions

    int w_, h_;   // frame width and height in pixels, fixed when the object is created

    EdgeFilter edgeFilter_ = EdgeFilter::Sobel;   // which edge filter process() runs; starts as Sobel

    // "= nullptr" gives each pointer a safe starting value (pointing at nothing) until the
    // constructor fills it in with cudaMalloc. cudaFree(nullptr) is harmless, so the
    // destructor is safe even if something was never allocated.

    // Sobel pipeline buffers
    uint8_t* d_bgr_   = nullptr;   // GPU copy of the colour frame (w*h*3 bytes)
    uint8_t* d_gray_  = nullptr;   // grayscale image (w*h bytes); also the input to the DCT
    uint8_t* d_edges_ = nullptr;   // edge image from Sobel or Laplacian (w*h bytes)

    // DCT buffers (K here means DCT_K)
    float* d_cosW_ = nullptr;   // W x K table: d_cosW_[x*K + u] = a_u cos(pi(2x+1)u / 2W)
    float* d_cosH_ = nullptr;   // H x K table: d_cosH_[y*K + v] = a_v cos(pi(2y+1)v / 2H)
                                // (the precomputed cosine values, with the scale factors a_u / a_v
                                //  already multiplied in; built once in the constructor)
    float* d_rows_ = nullptr;   // H x K: row-pass result G(u, y), stored [y*K + u]
                                // (output of DCT pass 1, input to pass 2)
    float* d_coef_ = nullptr;   // K x K: final coefficients, stored [v*K + u]
                                // (copied back to main's 'coef' vector; also the IDCT's input)

    // IDCT buffers
    float*   d_icols_ = nullptr;   // H x K: column-pass result, stored [y*K + u]
                                   // (output of IDCT pass 1, input to pass 2)
    uint8_t* d_recon_ = nullptr;   // W x H: rebuilt image (copied back to main's 'recon' image)

    // A CUDA stream: one queue that all of this object's GPU work goes into, so it runs
    // in order. {} sets it to an empty/zero value until the constructor creates it.
    cudaStream_t stream_{};
    // Timing markers: start_ | Sobel | sobelDone_ | DCT | dctDone_ | IDCT | stop_
    // Each event is a timestamp recorded on the GPU; the time between two neighbouring
    // events gives the time for the stage between them.
    cudaEvent_t start_{}, sobelDone_{}, dctDone_{}, stop_{};
    // The most recent timings in milliseconds, filled in at the end of process()
    // and read by main.cpp through the getters above.
    float lastMs_ = 0.0f;
    float lastDctMs_ = 0.0f;
    float lastIdctMs_ = 0.0f;
};   // <- a class definition must end with a semicolon
