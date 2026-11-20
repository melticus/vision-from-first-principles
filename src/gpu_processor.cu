
// =====================================================================================
// gpu_processor.cu — the CUDA (GPU) side of the webcam program.
//
// WHAT THIS FILE DOES (big picture)
// ---------------------------------
// It implements the GpuProcessor class that main() calls once per frame. Each call:
//   1. copies the colour camera frame from the CPU to the GPU,
//   2. converts it to grayscale            (kernel: bgrToGray),
//   3. finds edges with a Sobel or a Laplacian filter, whichever is selected
//                                          (kernel: sobel or laplacian),
//   4. computes the lowest K x K coefficients of the 2D DCT of the whole image
//                                          (kernels: dctRows then dctCols),
//   5. rebuilds a blurred image from only the lowest keep x keep coefficients
//                                          (kernels: idctCols then idctRows),
//   6. copies the results back to the CPU and measures how long each stage took.
//
// QUICK CUDA VOCABULARY
// ---------------------
//   host    = the CPU and its normal RAM
//   device  = the GPU and its own memory (VRAM). Pointers named d_... point to GPU memory.
//   kernel  = a function marked __global__ that runs on the GPU, many copies at once.
//   thread  = one copy of a kernel. Here, usually one thread per pixel (or per coefficient).
//   block   = a group of threads (here 16 x 16 = 256 threads) that run together.
//   grid    = all the blocks launched for one kernel call.
//   kernel<<<grid, block, sharedMem, stream>>>(args)  = how a kernel is launched.
//   stream  = a queue of GPU work. Things put in the same stream run in order.
//   event   = a timestamp marker dropped into a stream, used here for timing.
// =====================================================================================

#include "gpu_processor.cuh"   // Declaration of the GpuProcessor class (member variables, DCT_K, etc.)
#include <cmath>               // std::sqrt, std::cos (used on the CPU when building cosine tables)
#include <cstdio>              // std::fprintf for error messages
#include <cstdlib>             // std::exit to stop the program on an error
#include <vector>              // std::vector for the cosine tables built on the CPU

// ---- Error-checking macro -----------------------------------------------------------
// Almost every CUDA function returns a cudaError_t. This macro wraps a call so that if
// it fails, we print the error, the file name and the line number, then quit.
//   CUDA_CHECK(cudaMalloc(...));   ->  runs cudaMalloc, checks the result.
// The "\" at the end of each line continues the macro onto the next line.
// "do { ... } while (0)" is a standard trick so the macro behaves like a single
// statement (it works safely after an "if" without braces and needs a ";" after it).
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                /* run the call, keep result */ \
        if (err_ != cudaSuccess) {                /* anything but success = error */ \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                         cudaGetErrorString(err_), __FILE__, __LINE__);         \
                         /* error text, this file's name, this line number */   \
            std::exit(1);                         /* stop the program */          \
        }                                                                       \
    } while (0)

// Short name for the number of DCT coefficients per side (defined in the header).
// constexpr = a compile-time constant, so the GPU kernels can use it directly.
constexpr int K = GpuProcessor::DCT_K;

// ======================================================= Sobel kernels ====

// One thread per pixel: BGR -> grayscale (ITU-R BT.601 weights)
// __global__          = this function runs on the GPU and is launched from the CPU.
// const uint8_t*      = read-only array of bytes (0..255).
// __restrict__        = a promise to the compiler that these pointers don't overlap,
//                       which lets it optimise memory reads.
__global__ void bgrToGray(const uint8_t* __restrict__ bgr,     // input: colour image, 3 bytes per pixel
                          uint8_t* __restrict__ gray,          // output: grayscale image, 1 byte per pixel
                          int w, int h)                        // image width and height
{
    // Work out which pixel THIS thread is responsible for.
    // blockIdx  = which block this thread is in,   blockDim = size of a block (16 x 16),
    // threadIdx = position of the thread inside its block.
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // pixel column
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // pixel row
    if (x >= w || y >= h) return;   // The grid is rounded UP to whole blocks, so some threads
                                    // fall off the edge of the image. They just do nothing.

    int i = y * w + x;              // Convert (x, y) into a position in the flat 1D array
    const uint8_t* p = bgr + 3 * i; // Pointer to this pixel's 3 colour bytes.
                                    // OpenCV stores colours in the order Blue, Green, Red.
    // Weighted sum: green looks brightest to the eye, blue darkest (weights add up to 1).
    // p[0] = Blue, p[1] = Green, p[2] = Red. The float result is truncated to a byte.
    gray[i] = static_cast<uint8_t>(0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2]);
}

// One thread per pixel: 3x3 Sobel gradient magnitude
// Sobel measures how quickly brightness changes around a pixel. Large change = edge.
__global__ void sobel(const uint8_t* __restrict__ gray,   // input: grayscale image
                      uint8_t* __restrict__ out,          // output: edge strength image
                      int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // pixel column for this thread
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // pixel row for this thread
    if (x >= w || y >= h) return;                    // outside the image -> do nothing

    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        out[y * w + x] = 0;   // border pixels have no full neighbourhood
        return;               // (Sobel needs a pixel on every side, so set edges of the image to black)
    }

    // Small helper ("lambda"): g(xx, yy) returns the gray value at (xx, yy) as an int.
    // Using int (not uint8_t) lets the sums below go negative and above 255 without wrapping.
    // [&] means the lambda can use the variables around it (gray, w).
    auto g = [&](int xx, int yy) { return static_cast<int>(gray[yy * w + xx]); };

    // Horizontal gradient gx, using the kernel     -1  0 +1
    //                                              -2  0 +2
    //                                              -1  0 +1
    // = (right column) - (left column), with the middle row counted twice.
    // Large |gx| means a vertical edge (brightness changes left-to-right).
    int gx = -g(x - 1, y - 1) - 2 * g(x - 1, y) - g(x - 1, y + 1)
             + g(x + 1, y - 1) + 2 * g(x + 1, y) + g(x + 1, y + 1);
    // Vertical gradient gy, using the kernel       -1 -2 -1
    //                                               0  0  0
    //                                              +1 +2 +1
    // = (bottom row) - (top row). Large |gy| means a horizontal edge.
    int gy = -g(x - 1, y - 1) - 2 * g(x, y - 1) - g(x + 1, y - 1)
             + g(x - 1, y + 1) + 2 * g(x, y + 1) + g(x + 1, y + 1);

    // Edge strength = length of the gradient vector (gx, gy) = sqrt(gx² + gy²).
    // sqrtf is the fast single-precision (float) square root on the GPU.
    float mag = sqrtf(static_cast<float>(gx * gx + gy * gy));
    // The magnitude can be bigger than 255 (max is about 1442), so cap it at 255
    // with fminf before storing it as a byte. Otherwise it would wrap round to a dark value.
    out[y * w + x] = static_cast<uint8_t>(fminf(mag, 255.0f));
}

// One thread per pixel: 3x3 Laplacian (8-neighbour), absolute value
// The Laplacian measures the SECOND derivative of brightness: how much a pixel differs
// from the average of its neighbours. It is 0 on flat areas and on smooth ramps, and
// large where brightness bends sharply, i.e. at edges. Unlike Sobel it has no direction,
// so one kernel is enough (no separate gx and gy).
// Kernel used:     1  1  1
//                  1 -8  1
//                  1  1  1
// = (sum of the 8 neighbours) - 8 x (centre pixel).
__global__ void laplacian(const uint8_t* __restrict__ gray,   // input: grayscale image
                          uint8_t* __restrict__ out,          // output: edge strength image
                          int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // pixel column for this thread
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // pixel row for this thread
    if (x >= w || y >= h) return;                    // outside the image -> do nothing

    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        out[y * w + x] = 0;   // border pixels have no full neighbourhood, same as Sobel
        return;
    }

    // Same helper as in sobel(): gray value at (xx, yy) as an int, so sums can go negative.
    auto g = [&](int xx, int yy) { return static_cast<int>(gray[yy * w + xx]); };

    int lap = g(x - 1, y - 1) + g(x, y - 1) + g(x + 1, y - 1)   // row above
            + g(x - 1, y)                   + g(x + 1, y)       // left and right
            + g(x - 1, y + 1) + g(x, y + 1) + g(x + 1, y + 1)   // row below
            - 8 * g(x, y);                                      // minus 8 x centre

    // The result is negative on one side of an edge and positive on the other, so take
    // the absolute value (abs works on ints on the GPU too). It can reach 8 x 255 = 2040,
    // so cap it at 255 before storing it as a byte.
    out[y * w + x] = static_cast<uint8_t>(min(abs(lap), 255));
}

// ========================================================= DCT kernels ====
// Whole-image 2D DCT-II (orthonormal), lowest K x K coefficients only:
//   F(u,v) = a_u a_v sum_y sum_x f(x,y) cos(pi(2x+1)u / 2W) cos(pi(2y+1)v / 2H)
// computed in two separable passes. The a_u, a_v factors are folded into the tables.
//
// In words: F(u,v) says how much of a cosine pattern with u half-waves across and
// v half-waves down is in the image. F(0,0) (the "DC" term) is the average brightness;
// small u,v = slow, smooth changes; big u,v = fine detail.
//
// "Separable" means the 2D sum can be done as two 1D sums: first along every row
// (over x), then down every column (over y). That is much cheaper than doing the
// double sum directly for each (u,v).
//
// "Orthonormal" means the a_u, a_v scale factors are chosen so the transform keeps the
// image's energy the same and the inverse uses the exact same cosine values.
//
// Instead of calling cos() on the GPU millions of times per frame, the cos(...) values
// (already multiplied by a_u or a_v) are worked out once and stored in the tables
// cosW (for the width) and cosH (for the height). See makeCosTable below.

// All DCT/IDCT kernels use 16 x 16 thread blocks spread over as many blocks as needed,
// so K is not limited by the 1024-threads-per-block maximum.

// Pass 1, along each row:  G(u, y) = sum_x f(x, y) * cosW[x][u]
// One thread per (u, y). Neighbouring threads have neighbouring u, so they read
// neighbouring table entries cosW[x*K + u] together (coalesced).
// ("Coalesced" = threads next to each other read memory next to each other, so the GPU
//  can fetch it in one big read instead of many small ones. This matters a lot for speed.)
__global__ void dctRows(const uint8_t* __restrict__ gray,   // input: grayscale image (w x h)
                        const float* __restrict__ cosW,     // table: w rows x K columns
                        float* __restrict__ rows,           // output: h rows x K columns (the G values)
                        int w, int h)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;   // which horizontal frequency (0..K-1)
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // which image row (0..h-1)
    if (u >= K || y >= h) return;                    // extra threads from rounding up -> do nothing

    // Pointer to the start of image row y. size_t is used for the multiplication so it
    // can't overflow on very large images.
    const uint8_t* row = gray + static_cast<size_t>(y) * w;
    float sum = 0.0f;
    for (int x = 0; x < w; ++x)                               // walk along the whole row...
        sum += static_cast<float>(row[x]) * cosW[x * K + u];  // ...pixel value x cosine weight
        // All threads with the same y read the same row[x], which the GPU can share
        // (broadcast) between them.

    rows[y * K + u] = sum;   // Store G(u, y): "how much of frequency u is in row y"
}

// Pass 2, down each column:  F(u, v) = sum_y G(u, y) * cosH[y][v]
// One thread per (u, v).
// Takes the per-row results from pass 1 and combines them down the image to get the
// final 2D coefficient.
__global__ void dctCols(const float* __restrict__ rows,   // input: G(u, y) from pass 1 (h x K)
                        const float* __restrict__ cosH,   // table: h rows x K columns
                        float* __restrict__ coef,         // output: the K x K DCT coefficients
                        int h)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;   // horizontal frequency (0..K-1)
    int v = blockIdx.y * blockDim.y + threadIdx.y;   // vertical frequency (0..K-1)
    if (u >= K || v >= K) return;                    // outside the K x K square -> do nothing

    float sum = 0.0f;
    for (int y = 0; y < h; ++y)                         // walk down every image row...
        sum += rows[y * K + u] * cosH[y * K + v];       // ...G(u, y) x vertical cosine weight

    coef[v * K + u] = sum;   // Store F(u, v). Layout: row v, column u (matches main.cpp's coef[v*K + u])
}

// ======================================================== IDCT kernels ====
// Inverse of the above, using only the lowest keep x keep coefficients (keep <= K):
//   f(x,y) = sum_{v<keep} sum_{u<keep} a_u a_v F(u,v) cos(pi(2x+1)u / 2W) cos(pi(2y+1)v / 2H)
// Because the transform is orthonormal, the same cosine tables work in reverse.
// The missing high frequencies are treated as zero, so the result is a low-pass
// (blurred) version of the grayscale image; a larger keep gives a sharper image.
//
// The IDCT is also separable, so it's done in two passes again, this time in the
// opposite order (columns first, then rows) going from frequencies back to pixels.

// Pass 1, down each column:  C(u, y) = sum_{v<keep} F(u, v) * cosH[y][v]
// One thread per (u, y) with u < keep, each summing keep terms.
__global__ void idctCols(const float* __restrict__ coef,    // input: K x K DCT coefficients
                         const float* __restrict__ cosH,    // table: h rows x K columns
                         float* __restrict__ icols,         // output: h rows x K columns (only u < keep filled)
                         int h, int keep)                   // image height, how many coefficients to use
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;   // horizontal frequency (only 0..keep-1 used)
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // image row
    if (u >= keep || y >= h) return;                 // frequencies beyond keep are ignored (treated as 0)

    float sum = 0.0f;
    for (int v = 0; v < keep; ++v)                      // only the lowest 'keep' vertical frequencies
        sum += coef[v * K + u] * cosH[y * K + v];       // coefficient x vertical cosine at row y

    icols[y * K + u] = sum;   // C(u, y): frequency u's contribution, spread back over row y
}

// Pass 2, along each row:  f(x, y) = sum_{u<keep} C(u, y) * cosW[x][u]
// One thread per pixel, each summing keep terms; result rounded and clamped to 0..255.
__global__ void idctRows(const float* __restrict__ icols,   // input: C(u, y) from pass 1
                         const float* __restrict__ cosW,    // table: w rows x K columns
                         uint8_t* __restrict__ out,         // output: rebuilt grayscale image (w x h)
                         int w, int h, int keep)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // pixel column
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // pixel row
    if (x >= w || y >= h) return;                    // outside the image -> do nothing

    float sum = 0.0f;
    for (int u = 0; u < keep; ++u)                      // only the lowest 'keep' horizontal frequencies
        sum += icols[y * K + u] * cosW[x * K + u];      // C(u, y) x horizontal cosine at column x

    // Turn the float back into a pixel byte:
    //   + 0.5f      -> so the conversion rounds to nearest instead of always rounding down
    //   fmaxf(.., 0) -> anything negative becomes 0 (cutting off frequencies can overshoot)
    //   fminf(.., 255) -> anything above 255 becomes 255
    out[y * w + x] = static_cast<uint8_t>(fminf(fmaxf(sum + 0.5f, 0.0f), 255.0f));
}

// ========================================================== host class ====
// Everything below runs on the CPU ("host"). It sets up GPU memory and launches the
// kernels above.

// Table of n x K values: t[i*K + k] = a_k cos(pi(2i+1)k / 2n), normalisation folded in.
// Computed once in double precision on the CPU, then stored as float.
// n is either the image width (for cosW) or the image height (for cosH).
// 'static' here means this function is only visible inside this file.
static std::vector<float> makeCosTable(int n)
{
    std::vector<float> t(static_cast<size_t>(n) * K);   // n rows x K columns, stored flat
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < n; ++i) {             // i = pixel position (x or y)
        for (int k = 0; k < K; ++k) {         // k = frequency (u or v)
            // Orthonormal scale factor: sqrt(1/n) for the DC term (k = 0),
            // sqrt(2/n) for every other frequency.
            double a = (k == 0) ? std::sqrt(1.0 / n) : std::sqrt(2.0 / n);
            // Store a_k * cos(pi * (2i + 1) * k / (2n)), the DCT-II cosine basis value.
            // Worked out in double for accuracy, then saved as float (the GPU uses float).
            t[static_cast<size_t>(i) * K + k] =
                static_cast<float>(a * std::cos(pi * (2.0 * i + 1.0) * k / (2.0 * n)));
        }
    }
    return t;
}

// ---- Constructor: runs once (or whenever the frame size changes) --------------------
// ": w_(width), h_(height)" is a member initialiser list: it sets w_ and h_ before the body runs.
GpuProcessor::GpuProcessor(int width, int height) : w_(width), h_(height)
{
    // Checked when compiling: stop the build if DCT_K was set to something silly.
    static_assert(K >= 1 && K <= 256, "DCT_K must be between 1 and 256");
    // Checked when running: we can't take more frequencies than there are pixels.
    if (w_ < K || h_ < K) {
        std::fprintf(stderr, "GpuProcessor: image %dx%d is smaller than DCT_K=%d\n", w_, h_, K);
        std::exit(1);
    }

    size_t px = static_cast<size_t>(w_) * h_;   // total number of pixels

    // cudaMalloc(&ptr, bytes) reserves memory on the GPU and stores its address in ptr.

    // Sobel pipeline
    CUDA_CHECK(cudaMalloc(&d_bgr_, px * 3));    // colour input: 3 bytes per pixel
    CUDA_CHECK(cudaMalloc(&d_gray_, px));       // grayscale: 1 byte per pixel
    CUDA_CHECK(cudaMalloc(&d_edges_, px));      // Sobel output: 1 byte per pixel

    // DCT: cosine tables (built once) and working buffers
    std::vector<float> cosW = makeCosTable(w_);   // CPU copy of the width table  (w x K)
    std::vector<float> cosH = makeCosTable(h_);   // CPU copy of the height table (h x K)
    CUDA_CHECK(cudaMalloc(&d_cosW_, cosW.size() * sizeof(float)));   // GPU space for cosW
    CUDA_CHECK(cudaMalloc(&d_cosH_, cosH.size() * sizeof(float)));   // GPU space for cosH
    CUDA_CHECK(cudaMalloc(&d_rows_, static_cast<size_t>(h_) * K * sizeof(float)));  // pass-1 results G(u, y)
    CUDA_CHECK(cudaMalloc(&d_coef_, K * K * sizeof(float)));                         // final K x K coefficients
    // Copy the two tables from CPU memory to GPU memory. They never change, so this
    // happens once here instead of every frame.
    CUDA_CHECK(cudaMemcpy(d_cosW_, cosW.data(), cosW.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cosH_, cosH.data(), cosH.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    // IDCT
    CUDA_CHECK(cudaMalloc(&d_icols_, static_cast<size_t>(h_) * K * sizeof(float)));  // pass-1 results C(u, y)
    CUDA_CHECK(cudaMalloc(&d_recon_, px));                                           // rebuilt image, 1 byte/pixel

    // A stream to queue all the GPU work in order, and four events used as stopwatches:
    //   start_ -> sobelDone_ -> dctDone_ -> stop_
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&sobelDone_));
    CUDA_CHECK(cudaEventCreate(&dctDone_));
    CUDA_CHECK(cudaEventCreate(&stop_));
}

// ---- Destructor: runs automatically when the GpuProcessor is destroyed --------------
// Gives back every bit of GPU memory and every stream/event the constructor created.
// (No CUDA_CHECK here: if freeing fails during shutdown there's nothing useful to do.)
GpuProcessor::~GpuProcessor()
{
    cudaFree(d_bgr_);
    cudaFree(d_gray_);
    cudaFree(d_edges_);
    cudaFree(d_cosW_);
    cudaFree(d_cosH_);
    cudaFree(d_rows_);
    cudaFree(d_coef_);
    cudaFree(d_icols_);
    cudaFree(d_recon_);
    cudaStreamDestroy(stream_);
    cudaEventDestroy(start_);
    cudaEventDestroy(sobelDone_);
    cudaEventDestroy(dctDone_);
    cudaEventDestroy(stop_);
}

// ---- process(): called once per frame from main() -----------------------------------
//   bgr   - CPU pointer to the colour frame (input)
//   edges - CPU pointer where the Sobel image should be written (output)
//   dct   - CPU pointer for the K x K coefficients, or nullptr to skip copying them back
//   recon - CPU pointer for the rebuilt image, or nullptr to skip the IDCT
//   keep  - how many coefficients per side the IDCT should use
void GpuProcessor::process(const uint8_t* bgr, uint8_t* edges, float* dct, uint8_t* recon,
                           int keep)
{
    size_t px = static_cast<size_t>(w_) * h_;   // number of pixels
    dim3 block(16, 16);                         // 16 x 16 = 256 threads per block
    // Enough blocks to cover the whole image. (a + b - 1) / b is integer division
    // that rounds UP, e.g. 1280 / 16 = 80 exactly, 721 / 16 -> 46 not 45.
    dim3 grid((w_ + block.x - 1) / block.x, (h_ + block.y - 1) / block.y);

    // Grids for the frequency-domain kernels: K (or keep) along x, image rows or K along y
    // blocksFor(n) = how many 16-wide blocks are needed to cover n items (rounded up).
    auto blocksFor = [](int n) { return static_cast<unsigned>((n + 15) / 16); };
    // Force keep into the range 1..K (same as std::clamp, written with ?: operators).
    keep = (keep < 1) ? 1 : (keep > K ? K : keep);

    // Drop the "start" timestamp into the stream. Everything below is queued after it.
    CUDA_CHECK(cudaEventRecord(start_, stream_));

    // ---- Sobel: upload, grayscale, edges, download --------------------------
    // Copy the colour frame CPU -> GPU. "Async" = queued in the stream. (Because the
    // OpenCV image is ordinary, non-"pinned" memory, CUDA may still wait for this copy.)
    CUDA_CHECK(cudaMemcpyAsync(d_bgr_, bgr, px * 3, cudaMemcpyHostToDevice, stream_));
    // Launch: <<<grid of blocks, threads per block, 0 bytes of shared memory, our stream>>>
    bgrToGray<<<grid, block, 0, stream_>>>(d_bgr_, d_gray_, w_, h_);   // colour -> gray
    // gray -> edges, using whichever filter was chosen with setEdgeFilter() (keys 1 / 2).
    // Both kernels take the same inputs and write the same output buffer, so nothing
    // else in the pipeline needs to know which one ran.
    if (edgeFilter_ == EdgeFilter::Laplacian)
        laplacian<<<grid, block, 0, stream_>>>(d_gray_, d_edges_, w_, h_);
    else
        sobel<<<grid, block, 0, stream_>>>(d_gray_, d_edges_, w_, h_);
    // Kernel launches don't return an error code, so ask CUDA whether the launches failed.
    CUDA_CHECK(cudaGetLastError());
    // Copy the edge image GPU -> CPU (into main()'s 'edges' cv::Mat).
    CUDA_CHECK(cudaMemcpyAsync(edges, d_edges_, px, cudaMemcpyDeviceToHost, stream_));

    // Timestamp: the Sobel stage is finished. Note that this stage's time includes the
    // upload and the download, not just the two kernels.
    CUDA_CHECK(cudaEventRecord(sobelDone_, stream_));

    // ---- DCT of the grayscale image (already on the GPU) --------------------
    // Also needed when only the reconstruction is wanted.
    // (The IDCT needs the coefficients, so we compute the DCT if EITHER output is wanted.
    //  The grayscale image d_gray_ is reused, so no new upload is needed.)
    if (dct || recon) {
        dim3 rowGrid(blocksFor(K), blocksFor(h_));     // one thread per (u, y)
        dctRows<<<rowGrid, block, 0, stream_>>>(d_gray_, d_cosW_, d_rows_, w_, h_);   // pass 1: along rows

        dim3 colGrid(blocksFor(K), blocksFor(K));      // one thread per (u, v)
        dctCols<<<colGrid, block, 0, stream_>>>(d_rows_, d_cosH_, d_coef_, h_);       // pass 2: down columns
        CUDA_CHECK(cudaGetLastError());                // check both launches

        if (dct)   // only copy the K x K coefficients back to the CPU if main() asked for them
            CUDA_CHECK(cudaMemcpyAsync(dct, d_coef_, K * K * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream_));
    }

    // Timestamp: DCT stage finished (if the DCT was skipped, this is right after sobelDone_,
    // so its time comes out as roughly 0 ms).
    CUDA_CHECK(cudaEventRecord(dctDone_, stream_));

    // ---- IDCT: rebuild the image from the lowest keep x keep coefficients ----
    if (recon) {
        dim3 colGrid(blocksFor(keep), blocksFor(h_));  // one thread per (u < keep, y)
        idctCols<<<colGrid, block, 0, stream_>>>(d_coef_, d_cosH_, d_icols_, h_, keep);   // pass 1: columns
        idctRows<<<grid, block, 0, stream_>>>(d_icols_, d_cosW_, d_recon_, w_, h_, keep);  // pass 2: rows, one thread per pixel
        CUDA_CHECK(cudaGetLastError());                // check both launches

        // Copy the rebuilt image GPU -> CPU (into main()'s 'recon' cv::Mat).
        CUDA_CHECK(cudaMemcpyAsync(recon, d_recon_, px, cudaMemcpyDeviceToHost, stream_));
    }

    // Timestamp: all the work for this frame has been queued up to here.
    CUDA_CHECK(cudaEventRecord(stop_, stream_));

    // ---- Wait once for everything, then read the timers ---------------------
    // Until now the CPU has only QUEUED work. This line makes the CPU wait until the GPU
    // reaches the stop_ event, i.e. until everything (including the copies back to the
    // CPU) is done. After this, edges / dct / recon in main() are safe to use.
    CUDA_CHECK(cudaEventSynchronize(stop_));
    // Time between pairs of events, in milliseconds. These are what main() shows in the
    // status bar via lastGpuMs(), lastDctMs() and lastIdctMs().
    CUDA_CHECK(cudaEventElapsedTime(&lastMs_, start_, sobelDone_));        // upload + gray + Sobel + download
    CUDA_CHECK(cudaEventElapsedTime(&lastDctMs_, sobelDone_, dctDone_));   // DCT (+ copying coefficients back)
    CUDA_CHECK(cudaEventElapsedTime(&lastIdctMs_, dctDone_, stop_));       // IDCT (+ copying the image back)
}
