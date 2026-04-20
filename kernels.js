// ---------------------------------------------------------------
// KernelRule: binds a kernel to source/dest layers + growth params
// ---------------------------------------------------------------
class KernelRule {
	constructor(kernel, { source = 0, dest = 0, mu = 0.15, sigma = 0.015, weight = 1.0 } = {}) {
		this.kernel = kernel;
		this.source = source;
		this.dest = dest;
		this.mu = mu;
		this.sigma = sigma;
		this.weight = weight;
	}

	growth(u) {
		return 2 * Math.exp(-((u - this.mu) ** 2) / (2 * this.sigma * this.sigma)) - 1;
	}

	fmt_statusline() {
		return `[L${this.source}->L${this.dest}] ${this.kernel.fmt_statusline()} | mu:${this.mu} sigma:${this.sigma} w:${this.weight}`
	}
}

// All kernel core functions K_C(r) take r in [0, 1] (normalized distance)
// and return the unnormalized kernel weight at that distance.
// The compute() method handles the 2D distance calculation and normalization.

// Helper: shared compute logic for all kernels that use a 1D core function.
// core_fn(r) -> value for normalized distance r in [0, 1]
function _compute_kernel(kernel, core_fn) {
	let sum = 0;
	kernel.iter_elements((dx, dy) => {
		const dist = Math.sqrt((dx - kernel.half) ** 2 + (dy - kernel.half) ** 2) / kernel.half;

		if (dist > 1) {
			kernel.set(dx, dy, 0);
			return;
		}

		const value = core_fn(dist);
		kernel.set(dx, dy, value);
		sum += value;
	})

	// Normalise so kernel sums to 1
	if (sum > 0) {
		for (let i = 0; i < kernel.length; i++) {
			kernel[i] /= sum;
		}
	}
}

// ---------------------------------------------------------------
// Gaussian bump kernel (your original)
// K_C(r) = exp(-((r - peakRadius)^2) * beta)
// Not from the Wikipedia table, but a common simple choice.
// ---------------------------------------------------------------
class GaussianKernel extends KernelMatrix2D {
	constructor(size, peakRadius = 0.5, beta = 20) {
		super(size)
		this.peakRadius = peakRadius
		this.beta = beta
	}

	fmt_statusline() {
		return `GaussianKernel | beta:${this.beta} | peakRadius:${this.peakRadius}`
	}

	compute() {
		_compute_kernel(this, (r) => {
			return Math.exp(-((r - this.peakRadius) ** 2) * this.beta);
		})
	}
}

// ---------------------------------------------------------------
// Exponential kernel (Wikipedia)
// K_C(r) = exp(alpha - alpha / (4r(1-r))),  alpha = 4
// Smooth bump that is exactly 0 at r=0 and r=1.
// ---------------------------------------------------------------
class ExpKernel extends KernelMatrix2D {
	constructor(size, alpha = 4) {
		super(size)
		this.alpha = alpha
	}

	fmt_statusline() {
		return `ExpKernel | alpha:${this.alpha}`
	}

	compute() {
		_compute_kernel(this, (r) => {
			// Avoid division by zero at r=0 and r=1
			if (r <= 0 || r >= 1) return 0;
			return Math.exp(this.alpha - this.alpha / (4 * r * (1 - r)));
		})
	}
}

// ---------------------------------------------------------------
// Polynomial kernel (Wikipedia)
// K_C(r) = (4r(1-r))^alpha,  alpha = 4
// Similar ring shape, cheaper to compute than exponential.
// ---------------------------------------------------------------
class PolyKernel extends KernelMatrix2D {
	constructor(size, alpha = 4) {
		super(size)
		this.alpha = alpha
	}

	fmt_statusline() {
		return `PolyKernel | alpha:${this.alpha}`
	}

	compute() {
		_compute_kernel(this, (r) => {
			if (r <= 0 || r >= 1) return 0;
			return (4 * r * (1 - r)) ** this.alpha;
		})
	}
}

// ---------------------------------------------------------------
// Rectangular kernel (Wikipedia)
// K_C(r) = 1 if r in [1/4, 3/4], else 0
// Sharp ring with hard edges.
// ---------------------------------------------------------------
class RectKernel extends KernelMatrix2D {
	constructor(size, lo = 0.25, hi = 0.75) {
		super(size)
		this.lo = lo
		this.hi = hi
	}

	fmt_statusline() {
		return `RectKernel | range:[${this.lo}, ${this.hi}]`
	}

	compute() {
		_compute_kernel(this, (r) => {
			return (r >= this.lo && r <= this.hi) ? 1 : 0;
		})
	}
}

// Registry of available kernel types for the UI (must be after class definitions)
const KERNEL_TYPES = {
	'Gaussian':    { cls: GaussianKernel, defaults: { size: 31, peakRadius: 0.5, beta: 20 },
	                 create(p) { return new GaussianKernel(p.size, p.peakRadius, p.beta) } },
	'Exponential': { cls: ExpKernel, defaults: { size: 31, alpha: 4 },
	                 create(p) { return new ExpKernel(p.size, p.alpha) } },
	'Polynomial':  { cls: PolyKernel, defaults: { size: 31, alpha: 4 },
	                 create(p) { return new PolyKernel(p.size, p.alpha) } },
	'Rectangular': { cls: RectKernel, defaults: { size: 31, lo: 0.25, hi: 0.75 },
	                 create(p) { return new RectKernel(p.size, p.lo, p.hi) } },
};