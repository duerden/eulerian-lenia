document.addEventListener("DOMContentLoaded", async () => {
	var lenia = new Lenia("#meow");
})

class Lenia {
	canvas;
	ctx;

	world = null; //world!=null represents ready state
	n_steps = 0;

	world_width = 256;
	world_height = 256;
	canvas_width = 300;
	canvas_height = 150;
	layers = 1;

	// Offscreen canvas at world resolution for putImageData
	offscreen = null;
	offscreen_ctx = null;

	// Kernel rules: each binds a kernel to source->dest layer routing + growth params
	rules = [
		// Self-interactions (each layer drives itself)
		new KernelRule(new GaussianKernel(51, 0.5, 10), { source: 0, dest: 0, mu: 0.05, sigma: 0.003 }),
		//new KernelRule(new RectKernel(51, 0.1, 0.9), { source: 0, dest: 0, mu: 0.00, sigma: 0.003 }),
		//new KernelRule(new PolyKernel(41), { source: 1, dest: 1, mu: 0.01, sigma: 0.003 }),
		//new KernelRule(new RectKernel(11, 0.5, 20), { source: 1, dest: 1, mu: 0.01, sigma: 0.015 }),
		//new KernelRule(new ExpKernel(51, 0.5, 20), { source: 2, dest: 2, mu: 0.03, sigma: 0.015 }),
		//new KernelRule(new ExpKernel(51, 0.5, 20), { source: 0, dest: 0, mu: 0.03, sigma: 0.015 }),
		//new KernelRule(new ExpKernel(51, 0.5, 20), { source: 1, dest: 1, mu: 0.03, sigma: 0.015 }),

		// Cross-layer interactions
		//new KernelRule(new GaussianKernel(51, 0.1, 88), { source: 0, dest: 1, mu: 0.07, sigma: 0.015, weight: 1 }),
		//new KernelRule(new GaussianKernel(41, 0.5, 20), { source: 1, dest: 0, mu: 0.05, sigma: 0.015, weight: 1 }),
		//new KernelRule(new ExpKernel(15), { source: 0, dest: 2, mu: 0.20, sigma: 0.03, weight: 1 }),
		//new KernelRule(new PolyKernel(15), { source: 0, dest: 2, mu: 0.20, sigma: 0.03, weight: 1 }),

		//new KernelRule(new PolyKernel(55), { source: 2, dest: 1, mu: 0.02, sigma: 0.03, weight: 1 }),
		//new KernelRule(new RectKernel(71, 0.5, 20), { source: 2, dest: 1, mu: 0.01, sigma: 0.015, weight: 1 }),
	]

	// WASM state
	wasm = null; // the loaded module
	wasm_world_ptr = null;
	wasm_next_world_ptr = null;
	wasm_growth_ptr = null;
	wasm_kernel_ptrs = [];       // spatial kernels, one per rule
	wasm_kernel_fft_ptrs = [];   // FFT'd kernels, one per rule
	wasm_fft_temp_ptr = null;    // scratch buffer for FFT convolution (2*W*H)
	wasm_fft_col_buf_ptr = null; // scratch for fft2d column pass (2*max(W,H))
	wasm_layer_fft_ptrs = [];    // cached FFT of each source layer (one per layer, 2*W*H each)
	wasm_rgba_ptr = null;
	wasm_use_fft = false;        // true if world dims are power-of-2
	wasm_js_dirty = true;        // true when JS world[] is stale vs WASM

	running_raf = null;

	constructor(selector) {
		this.canvas = document.querySelectorAll(selector)[0];
		console.log("canvas", this.canvas)
		this.setup_canvas();

		//setup our kernels
		this.setup_kernels();

		//world dims are defined, construct world matrix
		this.setup_world();

		this.setup_gui();

		this.init_wasm();
	}

	is_ready() {return this.world != null}

	// ---- WASM integration ----

	async init_wasm() {
		if (typeof LeniaModule === "undefined") {
			console.warn("LeniaModule not found, using JS fallback for step/draw")
			return;
		}

		try {
			this.wasm = await LeniaModule();
			console.log("WASM module loaded")

			this._wasm_alloc_buffers();
			this._wasm_upload_world();
			this._wasm_upload_kernels();

			console.log("WASM buffers allocated and synced")

			// Initial draw now that WASM is ready
			this.draw();
		} catch(e) {
			console.error("Failed to init WASM, using JS fallback", e)
			this.wasm = null;
		}
	}

	_wasm_alloc_buffers() {
		const W = this.world_width, H = this.world_height, L = this.layers;
		const world_count = W * H * L;
		const rgba_count = W * H * 4;

		this.wasm_world_ptr = this.wasm._alloc_f64(world_count);
		this.wasm_next_world_ptr = this.wasm._alloc_f64(world_count);
		this.wasm_growth_ptr = this.wasm._alloc_f64(world_count);
		this.wasm_rgba_ptr = this.wasm._alloc_u8(rgba_count);

		// Check if we can use FFT (requires power-of-2 dimensions)
		this.wasm_use_fft = this.wasm._is_power_of_2(W) && this.wasm._is_power_of_2(H);
		if (this.wasm_use_fft) {
			// Scratch buffer for FFT convolution: interleaved complex, 2 * W * H doubles
			this.wasm_fft_temp_ptr = this.wasm._alloc_f64(2 * W * H);

			// Column scratch buffer for fft2d: 2 * max(W, H) doubles
			const max_dim = W > H ? W : H;
			this.wasm_fft_col_buf_ptr = this.wasm._alloc_f64(2 * max_dim);

			// Per-layer FFT cache buffers (one per layer, 2*W*H each)
			this.wasm_layer_fft_ptrs = [];
			for (let i = 0; i < L; i++) {
				this.wasm_layer_fft_ptrs.push(this.wasm._alloc_f64(2 * W * H));
			}

			console.log(`FFT convolution enabled (${W}x${H}, ${L} layers)`);
		} else {
			this.wasm_fft_temp_ptr = null;
			this.wasm_fft_col_buf_ptr = null;
			this.wasm_layer_fft_ptrs = [];
			console.log(`Spatial convolution (${W}x${H} is not power-of-2)`);
		}
	}

	_wasm_upload_world() {
		// Copy JS world data into WASM heap
		const f64view = new Float64Array(
			this.wasm.HEAPF64.buffer,
			this.wasm_world_ptr,
			this.world_width * this.world_height * this.layers
		);
		f64view.set(this.world);
	}

	_wasm_upload_kernels() {
		const W = this.world_width, H = this.world_height;

		// Free old kernel buffers
		for (const ptr of this.wasm_kernel_ptrs) this.wasm._free_f64(ptr);
		for (const ptr of this.wasm_kernel_fft_ptrs) this.wasm._free_f64(ptr);
		this.wasm_kernel_ptrs = [];
		this.wasm_kernel_fft_ptrs = [];

		for (const rule of this.rules) {
			const kernel = rule.kernel;
			const count = kernel.size * kernel.size;

			// Upload spatial kernel
			const ptr = this.wasm._alloc_f64(count);
			const f64view = new Float64Array(
				this.wasm.HEAPF64.buffer,
				ptr,
				count
			);
			f64view.set(kernel);
			this.wasm_kernel_ptrs.push(ptr);

			// Prepare FFT'd kernel if FFT is enabled
			if (this.wasm_use_fft) {
				const fft_ptr = this.wasm._alloc_f64(2 * W * H);
				this.wasm._fft_prepare_kernel(ptr, kernel.size, fft_ptr, W, H, this.wasm_fft_col_buf_ptr);
				this.wasm_kernel_fft_ptrs.push(fft_ptr);
			}
		}
	}

	_wasm_download_world() {
		// Copy WASM world state back to JS only when actually needed
		if (!this.wasm_js_dirty) return;
		const f64view = new Float64Array(
			this.wasm.HEAPF64.buffer,
			this.wasm_world_ptr,
			this.world_width * this.world_height * this.layers
		);
		this.world.set(f64view);
		this.wasm_js_dirty = false;
	}

	_wasm_free_buffers() {
		if (!this.wasm) return;
		if (this.wasm_world_ptr) this.wasm._free_f64(this.wasm_world_ptr);
		if (this.wasm_next_world_ptr) this.wasm._free_f64(this.wasm_next_world_ptr);
		if (this.wasm_growth_ptr) this.wasm._free_f64(this.wasm_growth_ptr);
		if (this.wasm_fft_temp_ptr) this.wasm._free_f64(this.wasm_fft_temp_ptr);
		if (this.wasm_fft_col_buf_ptr) this.wasm._free_f64(this.wasm_fft_col_buf_ptr);
		for (const ptr of this.wasm_layer_fft_ptrs) this.wasm._free_f64(ptr);
		for (const ptr of this.wasm_kernel_ptrs) this.wasm._free_f64(ptr);
		for (const ptr of this.wasm_kernel_fft_ptrs) this.wasm._free_f64(ptr);
		this.wasm_kernel_ptrs = [];
		this.wasm_kernel_fft_ptrs = [];
		this.wasm_layer_fft_ptrs = [];
		this.wasm_fft_temp_ptr = null;
		this.wasm_fft_col_buf_ptr = null;
		if (this.wasm_rgba_ptr) this.wasm._free_u8(this.wasm_rgba_ptr);
	}

	// ---- End WASM integration ----

	resize_canvas(){
		var width = Math.floor(window.innerWidth-0.01)
		var height = Math.floor(window.innerHeight-0.01)
		this.canvas.setAttribute("width", width)
		this.canvas.setAttribute("height", height)
		this.canvas_width = parseInt(this.canvas.getAttribute("width"))
		this.canvas_height = parseInt(this.canvas.getAttribute("height"))
		console.log("canvas resized to", `${this.canvas_width}*${this.canvas_height}`)
	}

	setup_offscreen(){
		this.offscreen = document.createElement("canvas");
		this.offscreen.width = this.world_width;
		this.offscreen.height = this.world_height;
		this.offscreen_ctx = this.offscreen.getContext("2d", { alpha: false });
	}

	resize_offscreen(){
		this.offscreen.width = this.world_width;
		this.offscreen.height = this.world_height;
	}

	setup_canvas(){
		this.resize_canvas()
		window.addEventListener("resize", () => this.resize_canvas())
		this.ctx = this.canvas.getContext("2d", {
			alpha: false,
			desynchronized: true,
		})
		this.setup_offscreen();

		// Click to seed a noise blob
		this.canvas.addEventListener("click", (e) => {
			if (!this.world) return;

			// Map canvas pixel to world coordinate
			const wx = Math.floor(e.offsetX / this.canvas_width * this.world_width);
			const wy = Math.floor(e.offsetY / this.canvas_height * this.world_height);

			// Ensure JS world is up to date before mutating
			if (this.wasm) this._wasm_download_world();

			for (let n = 0; n < this.layers; n++) {
				this.world.seed_blob(wx, wy, 25, n);
			}

			// Sync to WASM
			if (this.wasm) this._wasm_upload_world();

			this.draw();
		})

		// Right-click to erase a blob
		this.canvas.addEventListener("contextmenu", (e) => {
			e.preventDefault();
			if (!this.world) return;

			const wx = Math.floor(e.offsetX / this.canvas_width * this.world_width);
			const wy = Math.floor(e.offsetY / this.canvas_height * this.world_height);

			// Ensure JS world is up to date before mutating
			if (this.wasm) this._wasm_download_world();

			for (let n = 0; n < this.layers; n++) {
				this.world.erase_blob(wx, wy, 25, n);
			}

			if (this.wasm) this._wasm_upload_world();

			this.draw();
		})
	}

	gui = null;
	gui_rules_folder = null;
	gui_rule_folders = [];

	setup_gui(){
		const container = document.getElementById("gui-container");

		// ---- Tab bar ----
		const tab_bar = document.createElement('div');
		tab_bar.className = 'lenia-tabs';
		container.appendChild(tab_bar);

		const tab_panels = {};
		const tabs = {};
		const switch_tab = (name) => {
			for (const [k, panel] of Object.entries(tab_panels)) {
				panel.style.display = (k === name) ? '' : 'none';
				tabs[k].classList.toggle('active', k === name);
			}
		};

		const make_tab = (name) => {
			const btn = document.createElement('button');
			btn.textContent = name;
			btn.className = 'lenia-tab';
			btn.addEventListener('click', () => switch_tab(name));
			tab_bar.appendChild(btn);
			tabs[name] = btn;

			const panel = document.createElement('div');
			panel.style.display = 'none';
			container.appendChild(panel);
			tab_panels[name] = panel;
			return panel;
		};

		// ---- Create tabs ----
		const sim_panel = make_tab('Simulation');
		const presets_panel = make_tab('Presets');
		switch_tab('Simulation');

		// ---- Simulation tab (main GUI) ----
		this.gui = new lil.GUI({ container: sim_panel });
		this.gui.title('Lenia');

		// World settings
		const world_folder = this.gui.addFolder('World');
		this.gui_world_params = {
			width: this.world_width,
			height: this.world_height,
			layers: this.layers,
		};
		world_folder.add(this.gui_world_params, 'width', 16, 1024, 16).name('Width');
		world_folder.add(this.gui_world_params, 'height', 16, 1024, 16).name('Height');
		world_folder.add(this.gui_world_params, 'layers', 1, 8, 1).name('Layers');
		world_folder.add({ reset: () => this._gui_reset(this.gui_world_params) }, 'reset').name('Reset & Apply');

		// Simulation controls
		const sim_folder = this.gui.addFolder('Simulation');
		this.	gui_sim_state = { running: false, speed: 2.0, mass_conserved: false };
		sim_folder.add(this.gui_sim_state, 'speed', 0.1, 10, 0.1).name('Speed');
		sim_folder.add(this.gui_sim_state, 'mass_conserved').name('Mass Conservation');
		sim_folder.add({ step: () => this.step(0.1) }, 'step').name('Step');
		sim_folder.add({
			toggle: () => {
				if (this.running_raf == null) {
					let last_time = performance.now();
					const loop = (now) => {
						const dt = (now - last_time) / 1000;
						last_time = now;
						this.step(dt * this.gui_sim_state.speed);
						this.running_raf = requestAnimationFrame(loop);
					}
					this.running_raf = requestAnimationFrame(loop);
					this.gui_sim_state.running = true;
				} else {
					cancelAnimationFrame(this.running_raf);
					this.running_raf = null;
					this.gui_sim_state.running = false;
				}
			}
		}, 'toggle').name('Play / Pause');

		// Kernel rules
		this.gui_rules_folder = this.gui.addFolder('Kernel Rules');
		for (const rule of this.rules) {
			this._gui_add_rule_folder(rule);
		}

		const add_params = { type: Object.keys(KERNEL_TYPES)[0] };
		this.gui_rules_folder.add(add_params, 'type', Object.keys(KERNEL_TYPES)).name('Kernel Type');
		this.gui_rules_folder.add({
			add: () => {
				const typedef = KERNEL_TYPES[add_params.type];
				const kernel = typedef.create({ ...typedef.defaults });
				const rule = new KernelRule(kernel, {
					source: 0, dest: 0, mu: 0.15, sigma: 0.015, weight: 1.0
				});
				this._add_rule(rule);
			}
		}, 'add').name('+ Add Kernel Rule');

		// ---- Presets tab ----
		this.gui_presets = new lil.GUI({ container: presets_panel });
		this.gui_presets.title('Presets');
		this._setup_presets();
	}

	_setup_presets() {
		const presets = this._get_presets();
		for (const [name, preset] of Object.entries(presets)) {
			this.gui_presets.add({
				load: () => this._load_preset(preset)
			}, 'load').name(name);
		}
	}

	_load_preset(preset) {
		// Stop simulation
		if (this.running_raf != null) {
			cancelAnimationFrame(this.running_raf);
			this.running_raf = null;
		}

		// Apply world settings
		this.world_width = preset.width || 256;
		this.world_height = preset.height || 256;
		this.layers = preset.layers || 1;

		// Clear old rule GUI folders
		for (const f of this.gui_rule_folders) f.destroy();
		this.gui_rule_folders = [];

		// Set new rules
		this.rules = preset.rules.map(r => {
			const typedef = KERNEL_TYPES[r.kernel_type];
			const kernel = typedef.create(r.kernel_params);
			return new KernelRule(kernel, {
				source: r.source, dest: r.dest,
				mu: r.mu, sigma: r.sigma, weight: r.weight
			});
		});

		// Compute kernels
		this.setup_kernels();

		// Rebuild world
		this.resize_offscreen();
		this.setup_world();

		// Re-sync WASM
		if (this.wasm) {
			this._wasm_free_buffers();
			this._wasm_alloc_buffers();
			this._wasm_upload_world();
			this._wasm_upload_kernels();
		}

		// Rebuild rule GUI folders
		for (const rule of this.rules) {
			this._gui_add_rule_folder(rule);
		}

		// Update world params in GUI
		if (this.gui_world_params) {
			this.gui_world_params.width = this.world_width;
			this.gui_world_params.height = this.world_height;
			this.gui_world_params.layers = this.layers;
			this.gui.controllersRecursive().forEach(c => c.updateDisplay());
		}

		this.draw();
	}

	_get_presets() {
		return {
			// ===================================================================
			// EXISTING PRESETS
			// ===================================================================
			
			// 'Classic Lenia (1 layer)': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Gaussian', kernel_params: { size: 31, peakRadius: 0.5, beta: 20 },
			// 		  source: 0, dest: 0, mu: 0.15, sigma: 0.015, weight: 1.0 },
			// 	]
			// },
			// 'Sharp Rings': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Rectangular', kernel_params: { size: 31, lo: 0.25, hi: 0.75 },
			// 		  source: 0, dest: 0, mu: 0.15, sigma: 0.015, weight: 1.0 },
			// 	]
			// },
			// 'Smooth Life': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Polynomial', kernel_params: { size: 41, alpha: 4 },
			// 		  source: 0, dest: 0, mu: 0.10, sigma: 0.02, weight: 1.0 },
			// 	]
			// },
			// 'Two-Layer Symbiosis': {
			// 	width: 256, height: 256, layers: 2,
			// 	rules: [
			// 		{ kernel_type: 'Gaussian', kernel_params: { size: 31, peakRadius: 0.5, beta: 20 },
			// 		  source: 0, dest: 0, mu: 0.15, sigma: 0.015, weight: 1.0 },
			// 		{ kernel_type: 'Gaussian', kernel_params: { size: 31, peakRadius: 0.5, beta: 20 },
			// 		  source: 1, dest: 1, mu: 0.15, sigma: 0.015, weight: 1.0 },
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 21, alpha: 4 },
			// 		  source: 0, dest: 1, mu: 0.20, sigma: 0.03, weight: 0.5 },
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 21, alpha: 4 },
			// 		  source: 1, dest: 0, mu: 0.20, sigma: 0.03, weight: 0.5 },
			// 	]
			// },
			// 'RGB Cycle': {
			// 	width: 256, height: 256, layers: 3,
			// 	rules: [
			// 		{ kernel_type: 'Gaussian', kernel_params: { size: 31, peakRadius: 0.5, beta: 20 },
			// 		  source: 0, dest: 0, mu: 0.15, sigma: 0.015, weight: 1.0 },
			// 		{ kernel_type: 'Gaussian', kernel_params: { size: 31, peakRadius: 0.5, beta: 20 },
			// 		  source: 1, dest: 1, mu: 0.15, sigma: 0.015, weight: 1.0 },
			// 		{ kernel_type: 'Gaussian', kernel_params: { size: 31, peakRadius: 0.5, beta: 20 },
			// 		  source: 2, dest: 2, mu: 0.15, sigma: 0.015, weight: 1.0 },
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 15, alpha: 4 },
			// 		  source: 0, dest: 1, mu: 0.20, sigma: 0.03, weight: 0.5 },
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 15, alpha: 4 },
			// 		  source: 1, dest: 2, mu: 0.20, sigma: 0.03, weight: 0.5 },
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 15, alpha: 4 },
			// 		  source: 2, dest: 0, mu: 0.20, sigma: 0.03, weight: 0.5 },
			// 	]
			// },
			// 'Geminium (wide kernel)': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Gaussian', kernel_params: { size: 51, peakRadius: 0.5, beta: 10 },
			// 		  source: 0, dest: 0, mu: 0.14, sigma: 0.014, weight: 1.0 },
			// 	]
			// },
			// 'Multi-scale': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Gaussian', kernel_params: { size: 21, peakRadius: 0.5, beta: 20 },
			// 		  source: 0, dest: 0, mu: 0.15, sigma: 0.015, weight: 1.0 },
			// 		{ kernel_type: 'Gaussian', kernel_params: { size: 51, peakRadius: 0.5, beta: 20 },
			// 		  source: 0, dest: 0, mu: 0.12, sigma: 0.02, weight: 0.5 },
			// 	]
			// },

			// ===================================================================
			// ORIGINAL LENIA PAPER - "Lenia: Biology of Artificial Life" (Chan, 2019)
			//
			// Parameters from the paper: rank-1 species use beta=(1), exponential
			// kernel core with alpha=4, exponential growth mapping.
			// Key species parameters extracted from Figures 6, 7, 9, 15 and Tables 5-6.
			//
			// Paper kernel radius R maps to kernel size = 2*R+1.
			// Paper uses dt = 1/T where T is time resolution (typically T=10).
			// ===================================================================

			// --- Family Orbidae (O) - "disk bugs" ---
			// Orbium: the most studied species, mu=0.15, sigma=0.016, rank-1, beta=(1)
			// From Figure 6(a): R=13 (compact), or R~15 for smoother results
			// From Section 3.7: the showcase species used throughout the paper
			'Orbium (disk bug, classic lenia)': {
				width: 256, height: 256, layers: 1,
				rules: [
					{ kernel_type: 'Exponential', kernel_params: { size: 27, alpha: 4 },
					  source: 0, dest: 0, mu: 0.15, sigma: 0.016, weight: 1.0 },
				]
			},
			// // Orbium variant with slightly different params for faster movement
			// 'Orbium (fast)': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 31, alpha: 4 },
			// 		  source: 0, dest: 0, mu: 0.15, sigma: 0.02, weight: 1.0 },
			// 	]
			// },

			// // --- Family Scutidae (S) - "shield bugs" ---
			// // Scutium: disks with thick front, higher mu than Orbium
			// // From Figure 9: Scutium (S1) niche around mu=0.20-0.25, sigma=0.02-0.04
			// 'Scutium (shield bug)': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 31, alpha: 4 },
			// 		  source: 0, dest: 0, mu: 0.22, sigma: 0.03, weight: 1.0 },
			// 	]
			// },
			// // Discutium (S2): two fused scuta
			// // From Figure 9: S2 niche around mu=0.28-0.32, sigma=0.04-0.06
			// 'Discutium (double shield)': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 31, alpha: 4 },
			// 		  source: 0, dest: 0, mu: 0.30, sigma: 0.05, weight: 1.0 },
			// 	]
			// },
			// // Triscutium (S3): three fused scuta
			// // From Figure 9: S3 niche around mu=0.36, sigma=0.06-0.08
			// 'Triscutium (triple shield)': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 31, alpha: 4 },
			// 		  source: 0, dest: 0, mu: 0.36, sigma: 0.07, weight: 1.0 },
			// 	]
			// },

			// // --- Family Pterifera (P) - "winged bugs" ---
			// // Paraptera: one/two wings with sacs, rank-1 unit-4
			// // From Table 6 & Figure 15: mu=0.30 cross-section study
			// // P. arcus saliens: convex, jumping - sigma in [0.0468, 0.0515]
			// 'Paraptera (winged bug - jumping)': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 31, alpha: 4 },
			// 		  source: 0, dest: 0, mu: 0.30, sigma: 0.049, weight: 1.0 },
			// 	]
			// },
			// // P. cavus pedes: concave, walking - sigma in [0.0412, 0.0483]
			// 'Paraptera (winged bug - walking)': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 31, alpha: 4 },
			// 		  source: 0, dest: 0, mu: 0.30, sigma: 0.045, weight: 1.0 },
			// 	]
			// },
			// // P. sinus pedes: sinusoidal, deflected walking - sigma in [0.0404, 0.0414]
			// 'Paraptera (sinusoidal deflected)': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 31, alpha: 4 },
			// 		  source: 0, dest: 0, mu: 0.30, sigma: 0.041, weight: 1.0 },
			// 	]
			// },

			// // --- Family Helicidae (H) - "helix bugs" ---
			// // Rotating versions of Pterifera, higher mu values
			// // From Figure 9: H3-H7 niches at mu=0.33-0.42, sigma=0.015-0.04
			// 'Helicium (helix bug)': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 31, alpha: 4 },
			// 		  source: 0, dest: 0, mu: 0.35, sigma: 0.025, weight: 1.0 },
			// 	]
			// },

			// // --- Family Circidae (C) - "circle bugs" ---
			// // Circium: concentric rings, stationary, very low mu/sigma
			// // From Figure 9 inset: Circium (C1) at mu=0.14-0.16, sigma=0.01-0.015
			// 'Circium (ring bug)': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 31, alpha: 4 },
			// 		  source: 0, dest: 0, mu: 0.14, sigma: 0.012, weight: 1.0 },
			// 	]
			// },

			// --- Rank-1 "Primordial Soup" explorer ---
			// From Section 2.3.1: random generation strategy
			// Parameters in the middle of the class 4 "complex river" region
			'Primordial Soup (rank-1)': {
				width: 256, height: 256, layers: 1,
				rules: [
					{ kernel_type: 'Exponential', kernel_params: { size: 31, alpha: 4 },
					  source: 0, dest: 0, mu: 0.20, sigma: 0.03, weight: 1.0 },
				]
			},

			// // --- Rank-1 with polynomial kernel ---
			// // From Section 3.1.3: same species can exist with different core functions
			// // Polynomial version produces rougher but structurally similar patterns
			// 'Orbium (polynomial kernel)': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Polynomial', kernel_params: { size: 31, alpha: 4 },
			// 		  source: 0, dest: 0, mu: 0.15, sigma: 0.016, weight: 1.0 },
			// 	]
			// },

			// // --- Rank-1 with rectangular kernel (SmoothLife-like) ---
			// // From Section 3.1.3 & Figure 1(f): SmoothLife-like patterns
			// 'Orbium (rectangular/SmoothLife)': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Rectangular', kernel_params: { size: 31, lo: 0.25, hi: 0.75 },
			// 		  source: 0, dest: 0, mu: 0.15, sigma: 0.016, weight: 1.0 },
			// 	]
			// },

			// // --- Higher-rank approximation using multi-kernel ---
			// // The paper's rank-2 species use beta=(beta1, beta2), creating two concentric rings.
			// // We approximate this with two kernels at different radii.
			// // Pyroscutium (SP3): hybrid Scutium/Pterifera, rank-2 region
			// // From Figure 9: SP3 niche around mu=0.30, sigma=0.05
			// 'Pyroscutium (rank-2 approx)': {
			// 	width: 256, height: 256, layers: 1,
			// 	rules: [
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 31, alpha: 4 },
			// 		  source: 0, dest: 0, mu: 0.30, sigma: 0.05, weight: 1.0 },
			// 		{ kernel_type: 'Exponential', kernel_params: { size: 51, alpha: 4 },
			// 		  source: 0, dest: 0, mu: 0.25, sigma: 0.04, weight: 0.5 },
			// 	]
			// },

			// ===================================================================
			// EXPANDED UNIVERSE PAPER - "Lenia and Expanded Universe" (Chan, 2020)
			//
			// Extensions: multiple kernels, multiple channels (layers).
			// Paper formula: A_j^{t+dt} = [A_j^t + dt * sum_{i,k} (h_k/h) G_k(K_k * A_i^t)]
			// ===================================================================

			// --- Multi-kernel Lenia (2D, 1 channel, multiple kernels) ---
			// From Section "Multiple kernels": increases chaoticity but enables
			// higher self-organization, individuality, and self-replication.
			// Uses kernels at different scales with different growth params.
			'EU: Multi-kernel (self-replicator)': {
				width: 256, height: 256, layers: 1,
				rules: [
					{ kernel_type: 'Exponential', kernel_params: { size: 27, alpha: 4 },
					  source: 0, dest: 0, mu: 0.15, sigma: 0.016, weight: 1.0 },
					{ kernel_type: 'Exponential', kernel_params: { size: 41, alpha: 4 },
					  source: 0, dest: 0, mu: 0.32, sigma: 0.05, weight: 0.8 },
					{ kernel_type: 'Exponential', kernel_params: { size: 15, alpha: 4 },
					  source: 0, dest: 0, mu: 0.10, sigma: 0.01, weight: 0.3 },
				]
			},

			// --- Multi-kernel with individuality-promoting repulsion ---
			// Inspired by Figure 3(b): solitons with individuality and elastic collisions
			'EU: Multi-kernel (individuality)': {
				width: 256, height: 256, layers: 1,
				rules: [
					{ kernel_type: 'Exponential', kernel_params: { size: 31, alpha: 4 },
					  source: 0, dest: 0, mu: 0.18, sigma: 0.02, weight: 1.0 },
					{ kernel_type: 'Exponential', kernel_params: { size: 45, alpha: 4 },
					  source: 0, dest: 0, mu: 0.28, sigma: 0.04, weight: 0.6 },
				]
			},

			// --- Multi-channel "Aquarium" style ---
			// From Section "Differentiation": one genotype produces multiple phenotypes
			// Channels act as nucleus (ch0), membrane (ch1), motility (ch2)
			// Self-interacting + cross-channel kernels
			'EU: Aquarium (3-channel)': {
				width: 256, height: 256, layers: 3,
				rules: [
					// Self-interactions: each channel maintains its own patterns
					{ kernel_type: 'Exponential', kernel_params: { size: 27, alpha: 4 },
					  source: 0, dest: 0, mu: 0.15, sigma: 0.016, weight: 1.0 },
					{ kernel_type: 'Exponential', kernel_params: { size: 31, alpha: 4 },
					  source: 1, dest: 1, mu: 0.20, sigma: 0.03, weight: 1.0 },
					{ kernel_type: 'Exponential', kernel_params: { size: 25, alpha: 4 },
					  source: 2, dest: 2, mu: 0.12, sigma: 0.015, weight: 1.0 },
					// Cross-channel: ch0 (nucleus) drives ch1 (membrane)
					{ kernel_type: 'Exponential', kernel_params: { size: 21, alpha: 4 },
					  source: 0, dest: 1, mu: 0.18, sigma: 0.025, weight: 0.4 },
					// Cross-channel: ch1 (membrane) constrains ch2 (motility)
					{ kernel_type: 'Exponential', kernel_params: { size: 19, alpha: 4 },
					  source: 1, dest: 2, mu: 0.15, sigma: 0.02, weight: 0.3 },
					// Cross-channel: ch2 (motility) feeds back to ch0 (nucleus)
					{ kernel_type: 'Exponential', kernel_params: { size: 17, alpha: 4 },
					  source: 2, dest: 0, mu: 0.13, sigma: 0.015, weight: 0.2 },
				]
			},

			// --- Multi-channel with division of labor ---
			// From Section "Division of labor" & "Virtual cells":
			// ch0 = "nucleus" (central, drives replication)
			// ch1 = "cytoplasm/membrane" (defines boundaries, repulsion)
			'EU: Virtual Cell (2-channel)': {
				width: 256, height: 256, layers: 2,
				rules: [
					// Nucleus self-interaction (tighter kernel)
					{ kernel_type: 'Exponential', kernel_params: { size: 25, alpha: 4 },
					  source: 0, dest: 0, mu: 0.14, sigma: 0.014, weight: 1.0 },
					// Membrane self-interaction (wider kernel)
					{ kernel_type: 'Exponential', kernel_params: { size: 35, alpha: 4 },
					  source: 1, dest: 1, mu: 0.22, sigma: 0.035, weight: 1.0 },
					// Nucleus drives membrane formation
					{ kernel_type: 'Exponential', kernel_params: { size: 21, alpha: 4 },
					  source: 0, dest: 1, mu: 0.18, sigma: 0.025, weight: 0.5 },
					// Membrane stabilizes nucleus
					{ kernel_type: 'Exponential', kernel_params: { size: 19, alpha: 4 },
					  source: 1, dest: 0, mu: 0.16, sigma: 0.02, weight: 0.3 },
				]
			},

			// --- Multi-kernel chaotic/metamorphic ---
			// From Section "Common Phenomena > Locomotion": chaotic movements
			// and metamorphosis become more prevalent with multi-kernel
			'EU: Chaotic Metamorphosis': {
				width: 256, height: 256, layers: 1,
				rules: [
					{ kernel_type: 'Exponential', kernel_params: { size: 21, alpha: 4 },
					  source: 0, dest: 0, mu: 0.12, sigma: 0.01, weight: 1.0 },
					{ kernel_type: 'Exponential', kernel_params: { size: 35, alpha: 4 },
					  source: 0, dest: 0, mu: 0.25, sigma: 0.04, weight: 0.7 },
					{ kernel_type: 'Exponential', kernel_params: { size: 51, alpha: 4 },
					  source: 0, dest: 0, mu: 0.38, sigma: 0.06, weight: 0.4 },
				]
			},

			// --- Multi-channel emission patterns ---
			// From Section "Emission": puffer-train type emitters
			// Inspired by Figure 3(c): sea of emitted particles
			'EU: Emitter (3-channel)': {
				width: 256, height: 256, layers: 3,
				rules: [
					// Core pattern channel
					{ kernel_type: 'Exponential', kernel_params: { size: 27, alpha: 4 },
					  source: 0, dest: 0, mu: 0.16, sigma: 0.018, weight: 1.0 },
					// Emitted particle channel
					{ kernel_type: 'Exponential', kernel_params: { size: 15, alpha: 4 },
					  source: 1, dest: 1, mu: 0.10, sigma: 0.012, weight: 1.0 },
					// Signal channel
					{ kernel_type: 'Exponential', kernel_params: { size: 21, alpha: 4 },
					  source: 2, dest: 2, mu: 0.13, sigma: 0.015, weight: 0.8 },
					// Core emits particles
					{ kernel_type: 'Exponential', kernel_params: { size: 31, alpha: 4 },
					  source: 0, dest: 1, mu: 0.20, sigma: 0.03, weight: 0.5 },
					// Particles create signal trail
					{ kernel_type: 'Exponential', kernel_params: { size: 13, alpha: 4 },
					  source: 1, dest: 2, mu: 0.11, sigma: 0.015, weight: 0.3 },
					// Signal influences core
					{ kernel_type: 'Exponential', kernel_params: { size: 19, alpha: 4 },
					  source: 2, dest: 0, mu: 0.14, sigma: 0.018, weight: 0.2 },
				]
			},

			// --- Orbium elastic collision explorer ---
			// From Section 3.5.5 & Figure 12(k): particle reactions
			// Use Orbium params with slightly wider sigma for resilience
			'Orbium Collider': {
				width: 256, height: 256, layers: 1,
				rules: [
					{ kernel_type: 'Exponential', kernel_params: { size: 31, alpha: 4 },
					  source: 0, dest: 0, mu: 0.15, sigma: 0.018, weight: 1.0 },
				]
			},

			// --- Wider parameter space exploration ---
			// High mu/sigma region: larger, slower patterns (Scutidae/Helicidae territory)
			'Large Slow Patterns': {
				width: 256, height: 256, layers: 1,
				rules: [
					{ kernel_type: 'Exponential', kernel_params: { size: 41, alpha: 4 },
					  source: 0, dest: 0, mu: 0.40, sigma: 0.08, weight: 1.0 },
				]
			},

			// ===================================================================
			// LENIA4 WebGL SPECIES (Chan's multi-channel demo)
			// Translated from https://chakazul.github.io/Lenia/WebGL/
			// Uses Multi-Ring kernels to match the beta0/beta1/beta2 system.
			// R=12 -> kernel size 25, T=2 -> run at speed ~1
			// ===================================================================

			'Lenia4: Tri-Color Ghosts': {
				width: 256, height: 256, layers: 1,
				rules: [
					// Channel 0 self-interactions (3 rules)
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 25, rings: 2, betas: [0.25, 1] },
					  source: 0, dest: 0, mu: 0.16, sigma: 0.025, weight: 0.2 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 25, rings: 3, betas: [1, 0.75, 0.75] },
					  source: 0, dest: 0, mu: 0.22, sigma: 0.042, weight: 0.2 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 25, rings: 1, betas: [1] },
					  source: 0, dest: 0, mu: 0.28, sigma: 0.025, weight: 0.2 },
				]
			},

			'Lenia4: Tessellatium Fission': {
				width: 256, height: 256, layers: 3,
				rules: [
					// Adapted from VT049W (case 0) - first 9 self-interaction rules
					// src/dst: 0->0, 0->0, 0->0, 1->1, 1->1, 1->1, 2->2, 2->2, 2->2
					// Kernel sizes scaled by relR: 0.91, 0.62, 0.5, 0.72, 0.8, 0.96, 0.78, 0.79, 0.5
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 23, rings: 1, betas: [1] },
					  source: 0, dest: 0, mu: 0.272, sigma: 0.0595, weight: 0.19 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 17, rings: 1, betas: [1] },
					  source: 0, dest: 0, mu: 0.349, sigma: 0.1585, weight: 0.66 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 13, rings: 2, betas: [1, 0.25] },
					  source: 0, dest: 0, mu: 0.2, sigma: 0.0332, weight: 0.39 },
					// Channel 1 self
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 19, rings: 1, betas: [1] },
					  source: 1, dest: 1, mu: 0.447, sigma: 0.0777, weight: 0.74 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 21, rings: 2, betas: [5/6, 1] },
					  source: 1, dest: 1, mu: 0.247, sigma: 0.0342, weight: 0.92 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 25, rings: 1, betas: [1] },
					  source: 1, dest: 1, mu: 0.21, sigma: 0.0617, weight: 0.59 },
					// Channel 2 self
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 15, rings: 1, betas: [1] },
					  source: 2, dest: 2, mu: 0.462, sigma: 0.1192, weight: 0.37 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 21, rings: 1, betas: [1] },
					  source: 2, dest: 2, mu: 0.446, sigma: 0.1793, weight: 0.94 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 21, rings: 2, betas: [11/12, 1] },
					  source: 2, dest: 2, mu: 0.327, sigma: 0.1408, weight: 0.51 },
					// Cross-channel interactions (6 rules)
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 13, rings: 2, betas: [3/4, 1] },
					  source: 0, dest: 1, mu: 0.476, sigma: 0.0995, weight: 0.77 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 19, rings: 2, betas: [11/12, 1] },
					  source: 0, dest: 2, mu: 0.379, sigma: 0.0697, weight: 0.92 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 17, rings: 1, betas: [1] },
					  source: 1, dest: 0, mu: 0.262, sigma: 0.0877, weight: 0.71 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 15, rings: 2, betas: [1/6, 1] },
					  source: 1, dest: 2, mu: 0.412, sigma: 0.1101, weight: 0.59 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 21, rings: 1, betas: [1] },
					  source: 2, dest: 0, mu: 0.201, sigma: 0.0786, weight: 0.41 },
				]
			},

			'Lenia4: Zigzagging': {
				width: 256, height: 256, layers: 3,
				rules: [
					// Adapted from HAESRE (case 6)
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 23, rings: 1, betas: [1] },
					  source: 0, dest: 0, mu: 0.272, sigma: 0.0674, weight: 0.15 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 17, rings: 1, betas: [1] },
					  source: 0, dest: 0, mu: 0.337, sigma: 0.1576, weight: 0.474 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 17, rings: 2, betas: [1, 0.25] },
					  source: 0, dest: 0, mu: 0.129, sigma: 0.0382, weight: 0.342 },
					// Channel 1 self
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 25, rings: 2, betas: [0, 1] },
					  source: 1, dest: 1, mu: 0.132, sigma: 0.0514, weight: 0.192 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 19, rings: 1, betas: [1] },
					  source: 1, dest: 1, mu: 0.429, sigma: 0.0813, weight: 0.524 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 21, rings: 2, betas: [3/4, 1] },
					  source: 1, dest: 1, mu: 0.239, sigma: 0.0409, weight: 0.598 },
					// Channel 2 self
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 25, rings: 1, betas: [1] },
					  source: 2, dest: 2, mu: 0.25, sigma: 0.0691, weight: 0.426 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 19, rings: 1, betas: [1] },
					  source: 2, dest: 2, mu: 0.497, sigma: 0.1166, weight: 0.348 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 25, rings: 1, betas: [1] },
					  source: 2, dest: 2, mu: 0.486, sigma: 0.1751, weight: 0.62 },
					// Cross-channel
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 17, rings: 2, betas: [11/12, 1] },
					  source: 0, dest: 1, mu: 0.276, sigma: 0.1344, weight: 0.338 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 19, rings: 2, betas: [5/6, 1] },
					  source: 0, dest: 2, mu: 0.425, sigma: 0.1026, weight: 0.314 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 15, rings: 2, betas: [1, 11/12] },
					  source: 1, dest: 0, mu: 0.352, sigma: 0.0797, weight: 0.608 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 23, rings: 1, betas: [1] },
					  source: 1, dest: 2, mu: 0.21, sigma: 0.0921, weight: 0.292 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 21, rings: 2, betas: [1/4, 1] },
					  source: 2, dest: 0, mu: 0.381, sigma: 0.1056, weight: 0.426 },
					{ kernel_type: 'Multi-Ring', kernel_params: { size: 19, rings: 1, betas: [1] },
					  source: 2, dest: 1, mu: 0.244, sigma: 0.0813, weight: 0.346 },
				]
			},
		};
	}

	_gui_reset(world_params) {
		// Stop if running
		if (this.running_raf != null) {
			cancelAnimationFrame(this.running_raf);
			this.running_raf = null;
		}

		this.world_width = Math.floor(world_params.width);
		this.world_height = Math.floor(world_params.height);
		this.layers = Math.floor(world_params.layers);
		this.n_steps = 0;

		this.resize_offscreen();
		this.setup_world();

		if (this.wasm) {
			this._wasm_free_buffers();
			this._wasm_alloc_buffers();
			this._wasm_upload_world();
			this._wasm_upload_kernels();
		}

		this.draw();
	}

	_gui_add_rule_folder(rule) {
		const idx = this.rules.indexOf(rule);
		const folder = this.gui_rules_folder.addFolder(`Rule ${idx}: ${this._kernel_type_name(rule.kernel)} (L${rule.source} => L${rule.dest})`);

		// Layer routing
		folder.add(rule, 'source', 0, 7, 1).name('Source Layer');
		folder.add(rule, 'dest', 0, 7, 1).name('Dest Layer');

		// Growth params (with live preview update)
		const _update_growth = () => {
			if (folder._lenia_growth) this._draw_growth_preview(rule, folder._lenia_growth);
		};
		folder.add(rule, 'mu', 0, 0.5, 0.001).name('Mu').onChange(_update_growth);
		folder.add(rule, 'sigma', 0.001, 0.1, 0.001).name('Sigma').onChange(_update_growth);
		folder.add(rule, 'weight', -2, 2, 0.1).name('Weight').onChange(_update_growth);

		// Kernel params (depends on kernel type)
		const k_folder = folder.addFolder('Kernel Shape');
		const kernel = rule.kernel;
		const k_params = { size: kernel.size };

		k_folder.add(k_params, 'size', 3, 101, 2).name('Size').onChange(() => {
			k_params.size = Math.round((k_params.size - 1) / 2) * 2 + 1; // snap to odd
			this._rebuild_kernel(rule, k_params);
		});

		if (kernel instanceof GaussianKernel) {
			k_params.peakRadius = kernel.peakRadius;
			k_params.beta = kernel.beta;
			k_folder.add(k_params, 'peakRadius', 0, 1, 0.01).name('Peak Radius').onChange(() => {
				this._rebuild_kernel(rule, k_params);
			});
			k_folder.add(k_params, 'beta', 1, 100, 0.5).name('Beta').onChange(() => {
				this._rebuild_kernel(rule, k_params);
			});
		} else if (kernel instanceof ExpKernel) {
			k_params.alpha = kernel.alpha;
			k_folder.add(k_params, 'alpha', 0.5, 20, 0.5).name('Alpha').onChange(() => {
				this._rebuild_kernel(rule, k_params);
			});
		} else if (kernel instanceof PolyKernel) {
			k_params.alpha = kernel.alpha;
			k_folder.add(k_params, 'alpha', 0.5, 20, 0.5).name('Alpha').onChange(() => {
				this._rebuild_kernel(rule, k_params);
			});
		} else if (kernel instanceof RectKernel) {
			k_params.lo = kernel.lo;
			k_params.hi = kernel.hi;
			k_folder.add(k_params, 'lo', 0, 1, 0.01).name('Lo').onChange(() => {
				this._rebuild_kernel(rule, k_params);
			});
			k_folder.add(k_params, 'hi', 0, 1, 0.01).name('Hi').onChange(() => {
				this._rebuild_kernel(rule, k_params);
			});
		} else if (kernel instanceof CrossKernel) {
			k_params.armWidth = kernel.armWidth;
			k_params.beta = kernel.beta;
			k_folder.add(k_params, 'armWidth', 0.01, 0.5, 0.01).name('Arm Width').onChange(() => {
				this._rebuild_kernel(rule, k_params);
			});
			k_folder.add(k_params, 'beta', 0.5, 20, 0.5).name('Beta').onChange(() => {
				this._rebuild_kernel(rule, k_params);
			});
		} else if (kernel instanceof SpiralKernel) {
			k_params.arms = kernel.arms;
			k_params.tightness = kernel.tightness;
			k_params.armWidth = kernel.armWidth;
			k_folder.add(k_params, 'arms', 1, 6, 1).name('Arms').onChange(() => {
				this._rebuild_kernel(rule, k_params);
			});
			k_folder.add(k_params, 'tightness', 0.5, 10, 0.25).name('Tightness').onChange(() => {
				this._rebuild_kernel(rule, k_params);
			});
			k_folder.add(k_params, 'armWidth', 0.1, 2, 0.05).name('Arm Width').onChange(() => {
				this._rebuild_kernel(rule, k_params);
			});
		} else if (kernel instanceof MultiRingKernel) {
			k_params.rings = kernel.rings;
			k_params.betas = [...kernel.betas];
			// Ensure betas array has 3 slots for GUI
			while (k_params.betas.length < 3) k_params.betas.push(0);

			k_folder.add(k_params, 'rings', 1, 3, 1).name('Rings').onChange(() => {
				k_params.rings = Math.round(k_params.rings);
				this._rebuild_kernel(rule, k_params);
			});

			const beta_proxy = { beta0: k_params.betas[0], beta1: k_params.betas[1], beta2: k_params.betas[2] };
			const _update_betas = () => {
				k_params.betas = [beta_proxy.beta0, beta_proxy.beta1, beta_proxy.beta2];
				this._rebuild_kernel(rule, k_params);
			};
			k_folder.add(beta_proxy, 'beta0', 0, 1, 1/12).name('Ring 0 Height').onChange(_update_betas);
			k_folder.add(beta_proxy, 'beta1', 0, 1, 1/12).name('Ring 1 Height').onChange(_update_betas);
			k_folder.add(beta_proxy, 'beta2', 0, 1, 1/12).name('Ring 2 Height').onChange(_update_betas);
		}
		k_folder.close();

		// Preview container: kernel shape + growth curve side by side
		const preview_row = document.createElement('div');
		preview_row.style.cssText = 'display:flex;gap:4px;padding:4px;';

		const preview_canvas = document.createElement('canvas');
		preview_canvas.width = 64;
		preview_canvas.height = 64;
		preview_canvas.style.cssText = 'width:64px;height:64px;image-rendering:pixelated;border:1px solid #666;';

		const growth_canvas = document.createElement('canvas');
		growth_canvas.width = 128;
		growth_canvas.height = 64;
		growth_canvas.style.cssText = 'width:128px;height:64px;border:1px solid #666;flex:1;';

		preview_row.appendChild(preview_canvas);
		preview_row.appendChild(growth_canvas);
		folder.$children.appendChild(preview_row);

		// Draw initial previews
		this._draw_kernel_preview(rule.kernel, preview_canvas);
		this._draw_growth_preview(rule, growth_canvas);

		// Store references
		folder._lenia_preview = preview_canvas;
		folder._lenia_growth = growth_canvas;
		folder._lenia_rule = rule;
		folder._lenia_k_params = k_params;

		// Remove button
		folder.add({
			remove: () => {
				this._remove_rule(rule, folder);
			}
		}, 'remove').name('Remove Rule');

		folder.close();
		this.gui_rule_folders.push(folder);
		return folder;
	}

	_kernel_type_name(kernel) {
		if (kernel instanceof GaussianKernel) return 'Gaussian';
		if (kernel instanceof ExpKernel) return 'Exponential';
		if (kernel instanceof PolyKernel) return 'Polynomial';
		if (kernel instanceof RectKernel) return 'Rectangular';
		if (kernel instanceof SquareKernel) return 'Square';
		if (kernel instanceof CrossKernel) return 'Cross';
		if (kernel instanceof SpiralKernel) return 'Spiral';
		if (kernel instanceof MultiRingKernel) return 'Multi-Ring';
		return 'Unknown';
	}

	_rebuild_kernel(rule, k_params) {
		const type_name = this._kernel_type_name(rule.kernel);
		const typedef = KERNEL_TYPES[type_name];
		if (!typedef) return;

		rule.kernel = typedef.create(k_params);
		rule.kernel.compute();

		// Update preview
		const folder = this.gui_rule_folders.find(f => f._lenia_rule === rule);
		if (folder && folder._lenia_preview) {
			this._draw_kernel_preview(rule.kernel, folder._lenia_preview);
		}

		// Re-upload to WASM
		if (this.wasm) this._wasm_upload_kernels();
	}

	_draw_kernel_preview(kernel, canvas) {
		const ctx = canvas.getContext('2d');
		const img = ctx.createImageData(64, 64);
		const scale = kernel.size / 64;

		// Find max value for normalization
		let max_val = 0;
		for (let i = 0; i < kernel.length; i++) {
			if (kernel[i] > max_val) max_val = kernel[i];
		}
		if (max_val === 0) max_val = 1;

		for (let px = 0; px < 64; px++) {
			for (let py = 0; py < 64; py++) {
				const kx = Math.floor(px * scale);
				const ky = Math.floor(py * scale);
				let v = 0;
				if (kx < kernel.size && ky < kernel.size) {
					v = kernel.get(kx, ky) / max_val;
				}
				const i = (py * 64 + px) * 4;
				img.data[i + 0] = v * 255;
				img.data[i + 1] = v * 255;
				img.data[i + 2] = v * 255;
				img.data[i + 3] = 255;
			}
		}
		ctx.putImageData(img, 0, 0);
	}

	_draw_growth_preview(rule, canvas) {
		const W = canvas.width;
		const H = canvas.height;
		const ctx = canvas.getContext('2d');

		ctx.fillStyle = '#1a1a1a';
		ctx.fillRect(0, 0, W, H);

		const x_max = 0.5; // max convolution sum to display

		// Draw zero line (where growth = 0)
		const zero_y = H / 2;
		ctx.strokeStyle = '#444';
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(0, zero_y);
		ctx.lineTo(W, zero_y);
		ctx.stroke();

		// Draw mu marker (vertical dashed line)
		const mu_x = (rule.mu / x_max) * W;
		ctx.strokeStyle = '#888';
		ctx.setLineDash([3, 3]);
		ctx.beginPath();
		ctx.moveTo(mu_x, 0);
		ctx.lineTo(mu_x, H);
		ctx.stroke();
		ctx.setLineDash([]);

		// Draw growth curve
		// growth(u) ranges from -1 to +1, scaled by weight
		// y-axis: top = +max, bottom = -max
		const max_g = Math.max(Math.abs(rule.weight), 1);

		ctx.strokeStyle = '#4fc';
		ctx.lineWidth = 2;
		ctx.beginPath();
		for (let px = 0; px < W; px++) {
			const u = (px / W) * x_max;
			const g = rule.growth(u) * rule.weight;
			const y = zero_y - (g / max_g) * (H / 2 - 2);
			if (px === 0) ctx.moveTo(px, y);
			else ctx.lineTo(px, y);
		}
		ctx.stroke();

		// Labels
		ctx.fillStyle = '#888';
		ctx.font = '9px monospace';
		ctx.fillText('+' + max_g.toFixed(1), 2, 10);
		ctx.fillText('-' + max_g.toFixed(1), 2, H - 3);
		ctx.fillText('0', W - 10, zero_y - 3);
		ctx.fillText(x_max.toFixed(1), W - 20, H - 3);
	}

	_add_rule(rule) {
		rule.kernel.compute();
		this.rules.push(rule);
		this._gui_add_rule_folder(rule);

		if (this.wasm) this._wasm_upload_kernels();
	}

	_remove_rule(rule, folder) {
		const idx = this.rules.indexOf(rule);
		if (idx === -1) return;

		this.rules.splice(idx, 1);

		// Remove GUI folder
		const fi = this.gui_rule_folders.indexOf(folder);
		if (fi !== -1) this.gui_rule_folders.splice(fi, 1);
		folder.destroy();

		// Re-number remaining folder titles
		this.gui_rule_folders.forEach((f, i) => {
			f.title(`Rule ${i}: ${this._kernel_type_name(f._lenia_rule.kernel)}`);
		});

		if (this.wasm) this._wasm_upload_kernels();
	}

	setup_kernels(){
		this.rules.forEach(rule => rule.kernel.compute())
	}

	setup_world(){
		this.world = new WorldMatrix3D(this.world_width, this.world_height, this.layers)

		// Scatter random noise squares across each layer
		for (let n = 0; n < this.layers; n++) {
			this.world.seed_random_squares(12, 10, 25, n);
		}

		this.draw()
	}

	step(t){
		if (!this.wasm) return 0;
		const t0 = performance.now();

		this.n_steps = this.n_steps + 1;

		this._step_wasm(t);
		this.draw()

		const elapsed_ms = performance.now() - t0;
		return elapsed_ms;
	}

	_step_wasm(t) {
		const W = this.world_width, H = this.world_height;
		const world_count = W * H * this.layers;

		// 1. Clear growth accumulator
		this.wasm._clear_growth(this.wasm_growth_ptr, world_count);

		// 2. Apply each rule
		if (this.wasm_use_fft) {
			// --- FFT path with cached source layer transforms ---

			// Track which source layers have been FFT'd this step
			const fft_done = new Uint8Array(this.layers); // 0 = not yet

			for (let i = 0; i < this.rules.length; i++) {
				const rule = this.rules[i];
				const src = rule.source;

				// Forward-FFT the source layer if not already cached
				if (!fft_done[src]) {
					this.wasm._fft_forward_layer(
						this.wasm_world_ptr,
						src,
						this.wasm_layer_fft_ptrs[src],
						W, H,
						this.wasm_fft_col_buf_ptr
					);
					fft_done[src] = 1;
				}

				// Apply rule using cached source FFT
				this.wasm._fft_apply_rule(
					this.wasm_layer_fft_ptrs[src],
					this.wasm_growth_ptr,
					W, H,
					rule.dest,
					this.wasm_kernel_fft_ptrs[i],
					this.wasm_fft_temp_ptr,
					this.wasm_fft_col_buf_ptr,
					rule.mu,
					rule.sigma,
					rule.weight
				);
			}
		} else {
			// --- Spatial convolution fallback (non-power-of-2) ---
			for (let i = 0; i < this.rules.length; i++) {
				const rule = this.rules[i];
				this.wasm._apply_rule(
					this.wasm_world_ptr,
					this.wasm_growth_ptr,
					W, H,
					rule.source,
					rule.dest,
					this.wasm_kernel_ptrs[i],
					rule.kernel.size,
					rule.mu,
					rule.sigma,
					rule.weight
				);
			}
		}

		// 3. Apply accumulated growth
		if (this.gui_sim_state.mass_conserved) {
			this.wasm._apply_growth_conserved(
				this.wasm_world_ptr,
				this.wasm_next_world_ptr,
				this.wasm_growth_ptr,
				world_count,
				t
			);
		} else {
			this.wasm._apply_growth(
				this.wasm_world_ptr,
				this.wasm_next_world_ptr,
				this.wasm_growth_ptr,
				world_count,
				t
			);
		}

		// 4. Swap pointers (no memcpy!)
		const tmp = this.wasm_world_ptr;
		this.wasm_world_ptr = this.wasm_next_world_ptr;
		this.wasm_next_world_ptr = tmp;

		// Mark JS-side world as stale (download lazily on demand)
		this.wasm_js_dirty = true;
	}



	draw(){
		if (!this.wasm) return;
		this._draw_wasm();
	}

	_blit() {
		// Scale the world-sized offscreen canvas onto the visible canvas
		// Nearest-neighbor for crisp pixels when zoomed in
		this.ctx.imageSmoothingEnabled = false;
		this.ctx.clearRect(0, 0, this.canvas_width, this.canvas_height);
		this.ctx.drawImage(
			this.offscreen,
			0, 0, this.world_width, this.world_height,    // source
			0, 0, this.canvas_width, this.canvas_height    // dest (fills canvas)
		);
	}

	_draw_wasm() {
		this.wasm._draw(
			this.wasm_world_ptr,
			this.wasm_rgba_ptr,
			this.world_width,
			this.world_height,
			this.layers
		);

		// Write pixel data to offscreen canvas at world resolution
		const rgba_view = new Uint8ClampedArray(
			this.wasm.HEAPU8.buffer,
			this.wasm_rgba_ptr,
			this.world_width * this.world_height * 4
		);
		const image = new ImageData(rgba_view, this.world_width, this.world_height);
		this.offscreen_ctx.putImageData(image, 0, 0);

		// Scale offscreen onto visible canvas
		this._blit();
	}


}