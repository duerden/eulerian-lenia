#include <cmath>
#include <cstring>
#include <cstdint>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

// All buffers are allocated from JS and passed in as pointers.
// This keeps memory ownership on the JS side so kernels can be
// written directly from JS without any copying.

// ---------------------------------------------------------------
// FFT internals (not exported)
// ---------------------------------------------------------------

static const double PI = 3.14159265358979323846;

// In-place radix-2 Cooley-Tukey FFT on interleaved complex data.
// data: array of 2*n doubles, [re0, im0, re1, im1, ...]
// n: number of complex elements (must be power of 2)
// inverse: false = forward FFT, true = inverse FFT
static void fft1d(double* data, int n, bool inverse) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            // Swap complex elements i and j
            double tr = data[2*i];
            double ti = data[2*i+1];
            data[2*i]   = data[2*j];
            data[2*i+1] = data[2*j+1];
            data[2*j]   = tr;
            data[2*j+1] = ti;
        }
    }

    // Butterfly passes
    for (int len = 2; len <= n; len <<= 1) {
        double angle = 2.0 * PI / len * (inverse ? -1.0 : 1.0);
        double wr_step = cos(angle);
        double wi_step = sin(angle);

        for (int i = 0; i < n; i += len) {
            double wr = 1.0, wi = 0.0;
            for (int j = 0; j < len / 2; j++) {
                int u = i + j;
                int v = i + j + len / 2;
                double tr = data[2*v] * wr - data[2*v+1] * wi;
                double ti = data[2*v] * wi + data[2*v+1] * wr;
                data[2*v]   = data[2*u]   - tr;
                data[2*v+1] = data[2*u+1] - ti;
                data[2*u]   += tr;
                data[2*u+1] += ti;
                double wr_new = wr * wr_step - wi * wi_step;
                wi = wr * wi_step + wi * wr_step;
                wr = wr_new;
            }
        }
    }

    // Scale by 1/n for inverse
    if (inverse) {
        for (int i = 0; i < 2 * n; i++) {
            data[i] /= n;
        }
    }
}

// In-place 2D FFT on interleaved complex data.
// data: array of 2*W*H doubles, row-major, [re, im] interleaved
// W, H: dimensions (must be powers of 2)
// inverse: false = forward, true = inverse
static void fft2d(double* data, int W, int H, bool inverse) {
    // Temporary buffer for a single row/column
    int max_dim = W > H ? W : H;
    double* temp = new double[2 * max_dim];

    // Transform rows
    for (int y = 0; y < H; y++) {
        // Row y starts at data[2 * (y * W)]
        // Rows are already contiguous, apply 1D FFT in-place
        fft1d(&data[2 * y * W], W, inverse);
    }

    // Transform columns
    for (int x = 0; x < W; x++) {
        // Extract column x into temp
        for (int y = 0; y < H; y++) {
            temp[2*y]     = data[2 * (y * W + x)];
            temp[2*y + 1] = data[2 * (y * W + x) + 1];
        }
        fft1d(temp, H, inverse);
        // Write back
        for (int y = 0; y < H; y++) {
            data[2 * (y * W + x)]     = temp[2*y];
            data[2 * (y * W + x) + 1] = temp[2*y + 1];
        }
    }

    delete[] temp;
}

extern "C" {

// ---------------------------------------------------------------
// clear_growth()
// ---------------------------------------------------------------
EMSCRIPTEN_KEEPALIVE
void clear_growth(double* growth_acc, int count) {
    memset(growth_acc, 0, count * sizeof(double));
}

// ---------------------------------------------------------------
// apply_rule() - spatial convolution (fallback for non-power-of-2)
// ---------------------------------------------------------------
EMSCRIPTEN_KEEPALIVE
void apply_rule(
    double* world,
    double* growth_acc,
    int W,
    int H,
    int source_layer,
    int dest_layer,
    double* kernel,
    int kernel_size,
    double mu,
    double sigma,
    double weight
) {
    const int khalf = (kernel_size - 1) / 2;
    const int src_offset = source_layer * W * H;
    const int dst_offset = dest_layer * W * H;

    for (int x = 0; x < W; x++) {
        for (int y = 0; y < H; y++) {
            double sum = 0.0;
            for (int dx = 0; dx < kernel_size; dx++) {
                for (int dy = 0; dy < kernel_size; dy++) {
                    int nx = (x + dx - khalf + W) % W;
                    int ny = (y + dy - khalf + H) % H;
                    int ki = dx * kernel_size + dy;
                    sum += world[src_offset + nx * H + ny] * kernel[ki];
                }
            }
            double growth = 2.0 * exp(-((sum - mu) * (sum - mu)) / (2.0 * sigma * sigma)) - 1.0;
            growth_acc[dst_offset + x * H + y] += weight * growth;
        }
    }
}

// ---------------------------------------------------------------
// fft_prepare_kernel()
//
// Takes a spatial kernel (kernel_size x kernel_size) and produces
// its FFT padded to (W x H) world size, stored as interleaved
// complex doubles (2 * W * H doubles total).
//
// The kernel is placed centered at (0,0) with wrap-around, which
// is the correct layout for circular convolution matching the
// toroidal world boundary.
//
// kernel:        flat Float64Array [kernel_size * kernel_size]
// kernel_size:   diameter of the kernel (odd number)
// kernel_fft:    output buffer, 2 * W * H doubles (pre-allocated)
// W, H:          world dimensions (must be powers of 2)
// ---------------------------------------------------------------
EMSCRIPTEN_KEEPALIVE
void fft_prepare_kernel(
    double* kernel,
    int kernel_size,
    double* kernel_fft,
    int W,
    int H
) {
    const int khalf = (kernel_size - 1) / 2;
    const int N = W * H;

    // Zero the output (complex interleaved)
    memset(kernel_fft, 0, 2 * N * sizeof(double));

    // Place kernel centered at origin with wrap-around.
    // kernel[dx][dy] with dx,dy in [0, kernel_size) maps to
    // world position (dx - khalf, dy - khalf) with wrapping.
    // Our FFT layout is row-major: index = y * W + x (matching
    // the fft2d row/column pass order).
    for (int dx = 0; dx < kernel_size; dx++) {
        for (int dy = 0; dy < kernel_size; dy++) {
            int wx = (dx - khalf + W) % W;  // wrapped x
            int wy = (dy - khalf + H) % H;  // wrapped y
            int ki = dx * kernel_size + dy;
            // Row-major: index = wy * W + wx
            kernel_fft[2 * (wy * W + wx)] = kernel[ki];
            // Imaginary part stays 0
        }
    }

    // Forward FFT
    fft2d(kernel_fft, W, H, false);
}

// ---------------------------------------------------------------
// fft_apply_rule()
//
// FFT-based convolution for a single kernel rule.
// Convolves the source layer with a pre-FFT'd kernel, applies
// the growth function, and accumulates into the dest layer of
// growth_acc.
//
// world:         flat Float64Array [layers * W * H], layer-first,
//                stored as world[layer][x][y] = world[layer*W*H + x*H + y]
// growth_acc:    flat Float64Array [layers * W * H]
// W, H:          world dimensions (must be powers of 2)
// source_layer:  layer index to convolve
// dest_layer:    layer index for growth output
// kernel_fft:    pre-computed FFT of kernel (2 * W * H doubles)
// temp:          scratch buffer (2 * W * H doubles, pre-allocated)
// mu, sigma:     growth function parameters
// weight:        rule weight
// ---------------------------------------------------------------
EMSCRIPTEN_KEEPALIVE
void fft_apply_rule(
    double* world,
    double* growth_acc,
    int W,
    int H,
    int source_layer,
    int dest_layer,
    double* kernel_fft,
    double* temp,
    double mu,
    double sigma,
    double weight
) {
    const int N = W * H;
    const int src_offset = source_layer * W * H;
    const int dst_offset = dest_layer * W * H;

    // 1. Copy source layer into temp as complex data (row-major for FFT).
    //    World layout: world[src_offset + x*H + y] (column-major per layer)
    //    FFT layout: temp[2*(y*W + x)] (row-major)
    memset(temp, 0, 2 * N * sizeof(double));
    for (int x = 0; x < W; x++) {
        for (int y = 0; y < H; y++) {
            temp[2 * (y * W + x)] = world[src_offset + x * H + y];
        }
    }

    // 2. Forward FFT of source layer
    fft2d(temp, W, H, false);

    // 3. Pointwise complex multiplication: temp = temp * kernel_fft
    for (int i = 0; i < N; i++) {
        double ar = temp[2*i];
        double ai = temp[2*i + 1];
        double br = kernel_fft[2*i];
        double bi = kernel_fft[2*i + 1];
        temp[2*i]     = ar * br - ai * bi;
        temp[2*i + 1] = ar * bi + ai * br;
    }

    // 4. Inverse FFT
    fft2d(temp, W, H, true);

    // 5. Apply growth function and accumulate into growth_acc.
    //    Read back from row-major FFT layout to column-major world layout.
    const double inv_2ss = 1.0 / (2.0 * sigma * sigma);
    for (int x = 0; x < W; x++) {
        for (int y = 0; y < H; y++) {
            double conv = temp[2 * (y * W + x)]; // real part = convolution result
            double diff = conv - mu;
            double growth = 2.0 * exp(-(diff * diff) * inv_2ss) - 1.0;
            growth_acc[dst_offset + x * H + y] += weight * growth;
        }
    }
}

// ---------------------------------------------------------------
// apply_growth()
// ---------------------------------------------------------------
EMSCRIPTEN_KEEPALIVE
void apply_growth(
    double* world,
    double* next_world,
    double* growth_acc,
    int count,
    double dt
) {
    for (int i = 0; i < count; i++) {
        double val = world[i] + dt * growth_acc[i];
        if (val < 0.0) val = 0.0;
        if (val > 1.0) val = 1.0;
        next_world[i] = val;
    }
}

// ---------------------------------------------------------------
// apply_growth_conserved()
//
// Integrates growth into next_world while preserving total mass.
// Computes the target mass from the current world, applies growth
// with clamping, then iteratively redistributes any mass error
// among cells that have headroom (not pinned at 0 or 1).
//
// This avoids the problem where naive normalize-then-clamp leaks
// mass through the clamp boundaries.
// ---------------------------------------------------------------
EMSCRIPTEN_KEEPALIVE
void apply_growth_conserved(
    double* world,
    double* next_world,
    double* growth_acc,
    int count,
    double dt
) {
    // Compute target mass (sum of current world)
    double target_mass = 0.0;
    for (int i = 0; i < count; i++) {
        target_mass += world[i];
    }

    // Initial integration with clamping
    for (int i = 0; i < count; i++) {
        double val = world[i] + dt * growth_acc[i];
        if (val < 0.0) val = 0.0;
        if (val > 1.0) val = 1.0;
        next_world[i] = val;
    }

    // Iteratively redistribute mass error.
    // Each pass spreads the error among cells that aren't pinned
    // at a clamp boundary. Converges quickly (typically 2-4 passes).
    const int MAX_ITERS = 10;
    const double TOLERANCE = 1e-6;

    for (int iter = 0; iter < MAX_ITERS; iter++) {
        double current_mass = 0.0;
        for (int i = 0; i < count; i++) {
            current_mass += next_world[i];
        }

        double error = current_mass - target_mass;
        if (error > -TOLERANCE && error < TOLERANCE) break;

        // Count cells that can absorb the correction:
        // If error > 0 (too much mass), we need to subtract, so
        //   eligible cells are those with next_world[i] > 0.
        // If error < 0 (too little mass), we need to add, so
        //   eligible cells are those with next_world[i] < 1.
        int eligible = 0;
        for (int i = 0; i < count; i++) {
            if (error > 0 && next_world[i] > 0.0) eligible++;
            else if (error < 0 && next_world[i] < 1.0) eligible++;
        }
        if (eligible == 0) break;

        double correction = error / (double)eligible;

        for (int i = 0; i < count; i++) {
            if (error > 0 && next_world[i] > 0.0) {
                next_world[i] -= correction;
                if (next_world[i] < 0.0) next_world[i] = 0.0;
            } else if (error < 0 && next_world[i] < 1.0) {
                next_world[i] -= correction; // correction is negative
                if (next_world[i] > 1.0) next_world[i] = 1.0;
            }
        }
    }
}

// ---------------------------------------------------------------
// draw()
// ---------------------------------------------------------------
EMSCRIPTEN_KEEPALIVE
void draw(
    double* world,
    uint8_t* rgba,
    int W,
    int H,
    int n_layers
) {
    const int layer_size = W * H;

    for (int x = 0; x < W; x++) {
        for (int y = 0; y < H; y++) {
            const int pixel_idx = (y * W + x) * 4;
            const int world_idx = x * H + y;

            double r = (n_layers > 0) ? world[0 * layer_size + world_idx] : 0.0;
            double g = (n_layers > 1) ? world[1 * layer_size + world_idx] : 0.0;
            double b = (n_layers > 2) ? world[2 * layer_size + world_idx] : 0.0;

            rgba[pixel_idx + 0] = (uint8_t)(r * 255.0);
            rgba[pixel_idx + 1] = (uint8_t)(g * 255.0);
            rgba[pixel_idx + 2] = (uint8_t)(b * 255.0);
            rgba[pixel_idx + 3] = 255;
        }
    }
}

// ---------------------------------------------------------------
// is_power_of_2()
// ---------------------------------------------------------------
EMSCRIPTEN_KEEPALIVE
int is_power_of_2(int n) {
    return (n > 0) && ((n & (n - 1)) == 0) ? 1 : 0;
}

// ---------------------------------------------------------------
// Utility: allocate/free buffers from JS
// ---------------------------------------------------------------
EMSCRIPTEN_KEEPALIVE
double* alloc_f64(int count) {
    return new double[count];
}

EMSCRIPTEN_KEEPALIVE
uint8_t* alloc_u8(int count) {
    return new uint8_t[count];
}

EMSCRIPTEN_KEEPALIVE
void free_f64(double* ptr) {
    delete[] ptr;
}

EMSCRIPTEN_KEEPALIVE
void free_u8(uint8_t* ptr) {
    delete[] ptr;
}

} // extern "C"
