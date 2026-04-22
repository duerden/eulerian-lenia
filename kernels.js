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

// ---------------------------------------------------------------
// Square kernel
// K(x, y) = 1 for all cells (uniform box).
// Normalised so the kernel sums to 1.
// ---------------------------------------------------------------
class SquareKernel extends KernelMatrix2D {
	constructor(size) {
		super(size)
	}

	fmt_statusline() {
		return `SquareKernel`
	}

	compute() {
		const val = 1.0 / (this.size * this.size);
		for (let i = 0; i < this.length; i++) {
			this[i] = val;
		}
	}
}

// ---------------------------------------------------------------
// Cross kernel
// Weight concentrated along horizontal and vertical arms through
// the center, with Gaussian falloff from the arm centerline and
// from the kernel center outward.
// armWidth: controls how thick the arms are (in normalised units)
// beta: controls radial falloff from center
// ---------------------------------------------------------------
class CrossKernel extends KernelMatrix2D {
	constructor(size, armWidth = 0.15, beta = 4) {
		super(size)
		this.armWidth = armWidth
		this.beta = beta
	}

	fmt_statusline() {
		return `CrossKernel | armWidth:${this.armWidth} beta:${this.beta}`
	}

	compute() {
		let sum = 0;
		this.iter_elements((dx, dy) => {
			const nx = (dx - this.half) / this.half; // normalised to [-1, 1]
			const ny = (dy - this.half) / this.half;
			const r = Math.sqrt(nx * nx + ny * ny);

			if (r > 1) {
				this.set(dx, dy, 0);
				return;
			}

			// Distance from nearest arm (horizontal arm = y axis, vertical arm = x axis)
			const arm_dist = Math.min(Math.abs(nx), Math.abs(ny));

			// Gaussian falloff from arm centerline
			const arm_val = Math.exp(-(arm_dist * arm_dist) / (2 * this.armWidth * this.armWidth));
			// Radial falloff from center
			const radial_val = Math.exp(-r * r * this.beta);

			// Combine: strong near arms, fading outward
			const value = arm_val * (1 - radial_val) + radial_val;
			this.set(dx, dy, value);
			sum += value;
		})

		if (sum > 0) {
			for (let i = 0; i < this.length; i++) this[i] /= sum;
		}
	}
}

// ---------------------------------------------------------------
// Spiral kernel
// Weight follows a spiral arm from center outward.
// Breaks mirror symmetry — only has rotational symmetry.
// arms: number of spiral arms
// tightness: how tightly wound the spiral is
// armWidth: angular width of each arm (in radians)
// ---------------------------------------------------------------
class SpiralKernel extends KernelMatrix2D {
	constructor(size, arms = 1, tightness = 3, armWidth = 0.8) {
		super(size)
		this.arms = arms
		this.tightness = tightness
		this.armWidth = armWidth
	}

	fmt_statusline() {
		return `SpiralKernel | arms:${this.arms} tightness:${this.tightness}`
	}

	compute() {
		let sum = 0;
		this.iter_elements((dx, dy) => {
			const nx = (dx - this.half) / this.half;
			const ny = (dy - this.half) / this.half;
			const r = Math.sqrt(nx * nx + ny * ny);

			if (r > 1 || r < 0.01) {
				this.set(dx, dy, 0);
				return;
			}

			const angle = Math.atan2(ny, nx); // [-PI, PI]

			// Expected angle for a spiral arm at this radius
			// angle_on_spiral = tightness * r * 2*PI
			// We check distance to nearest arm
			let min_angular_dist = Infinity;
			for (let a = 0; a < this.arms; a++) {
				const arm_angle = this.tightness * r * 2 * Math.PI + (a * 2 * Math.PI / this.arms);
				// Angular distance (wrapped to [-PI, PI])
				let diff = angle - arm_angle;
				diff = diff - Math.round(diff / (2 * Math.PI)) * 2 * Math.PI;
				if (Math.abs(diff) < min_angular_dist) min_angular_dist = Math.abs(diff);
			}

			// Gaussian falloff from spiral arm centerline
			const arm_val = Math.exp(-(min_angular_dist * min_angular_dist) / (2 * this.armWidth * this.armWidth));

			// Smooth radial envelope: fade in from center, fade out at edge
			const envelope = Math.sin(r * Math.PI);

			const value = arm_val * envelope;
			this.set(dx, dy, value);
			sum += value;
		})

		if (sum > 0) {
			for (let i = 0; i < this.length; i++) this[i] /= sum;
		}
	}
}

// ---------------------------------------------------------------
// Multi-ring kernel (matching Lenia4 / Chan's WebGL demo)
// Radius is divided into `rings` equal segments. Each segment has
// a Gaussian bump (mu=0.5, sigma=0.15 in segment-local space),
// scaled by a per-ring height (beta array).
//
// rings: number of concentric rings (1-3)
// betas: array of ring heights, e.g. [1, 0.25] for 2 rings
//
// This matches the kernel formula from the WebGL demo:
//   Br = rings * r  (scaled radius)
//   ring_index = floor(Br)
//   height = betas[ring_index]
//   bump = exp(-((fract(Br) - 0.5)^2) / (2 * 0.15^2))
//   K(r) = height * bump
// ---------------------------------------------------------------
class MultiRingKernel extends KernelMatrix2D {
	constructor(size, rings = 1, betas = [1]) {
		super(size)
		this.rings = rings
		// Pad or trim betas to match ring count
		this.betas = Array.from({ length: rings }, (_, i) => (betas[i] !== undefined ? betas[i] : 0));
	}

	fmt_statusline() {
		return `MultiRingKernel | rings:${this.rings} betas:[${this.betas.map(b => b.toFixed(2)).join(',')}]`
	}

	compute() {
		const RING_MU = 0.5;
		const RING_SIGMA = 0.15;
		let sum = 0;

		this.iter_elements((dx, dy) => {
			const r = Math.sqrt((dx - this.half) ** 2 + (dy - this.half) ** 2) / this.half;

			if (r > 1 || r < 0) {
				this.set(dx, dy, 0);
				return;
			}

			const Br = this.rings * r;
			const ring_index = Math.min(Math.floor(Br), this.rings - 1);
			const height = this.betas[ring_index];
			const frac = Br - Math.floor(Br);
			const bump = Math.exp(-((frac - RING_MU) ** 2) / (2 * RING_SIGMA * RING_SIGMA));

			const value = height * bump;
			this.set(dx, dy, value);
			sum += value;
		})

		if (sum > 0) {
			for (let i = 0; i < this.length; i++) this[i] /= sum;
		}
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
	'Square':      { cls: SquareKernel, defaults: { size: 31 },
	                 create(p) { return new SquareKernel(p.size) } },
	'Cross':       { cls: CrossKernel, defaults: { size: 31, armWidth: 0.15, beta: 4 },
	                 create(p) { return new CrossKernel(p.size, p.armWidth, p.beta) } },
	'Spiral':      { cls: SpiralKernel, defaults: { size: 31, arms: 1, tightness: 3, armWidth: 0.8 },
	                 create(p) { return new SpiralKernel(p.size, p.arms, p.tightness, p.armWidth) } },
	'Multi-Ring':  { cls: MultiRingKernel, defaults: { size: 25, rings: 1, betas: [1] },
	                 create(p) { return new MultiRingKernel(p.size, p.rings, p.betas || [1]) } },
};