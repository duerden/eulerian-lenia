
class WorldMatrix3D extends Float64Array {
	size_x = 1; //width
	size_y = 1; //height
	size_l = 1; //layer

	constructor(sx,sy,sl){
		super(sx*sy*sl)
		this.size_x = sx
		this.size_y = sy
		this.size_l = sl
	}

	idx(x,y,l) {
		return l * this.size_x * this.size_y + x * this.size_y + y;
	}

	get(x,y,l) {
		return this[this.idx(x,y,l)]
	}

	get_layer_offset(layer_l){
		return layer_l * this.size_x * this.size_y 
	}

	seed_blob(cx, cy, radius, layer = 0) {
		const offset = this.get_layer_offset(layer);
		for (let x = 0; x < this.size_x; x++) {
			for (let y = 0; y < this.size_y; y++) {
				const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
				if (dist < radius) {
					const falloff = Math.max(0, 1 - (dist / radius) ** 2);
					this[offset + x * this.size_y + y] += Math.random() * falloff;
				}
			}
		}
	}

	erase_blob(cx, cy, radius, layer = 0) {
		const offset = this.get_layer_offset(layer);
		for (let x = 0; x < this.size_x; x++) {
			for (let y = 0; y < this.size_y; y++) {
				const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
				if (dist < radius) {
					const falloff = Math.max(0, 1 - (dist / radius) ** 2);
					this[offset + x * this.size_y + y] = Math.max(0, this[offset + x * this.size_y + y] - falloff);
				}
			}
		}
	}

	// Place a square of random noise at (cx, cy) on the given layer
	seed_square(cx, cy, size, layer = 0) {
		const offset = this.get_layer_offset(layer);
		const half = Math.floor(size / 2);
		for (let dx = -half; dx < half; dx++) {
			for (let dy = -half; dy < half; dy++) {
				const x = ((cx + dx) % this.size_x + this.size_x) % this.size_x;
				const y = ((cy + dy) % this.size_y + this.size_y) % this.size_y;
				this[offset + x * this.size_y + y] = Math.random();
			}
		}
	}

    seed_solid_blob(cx, cy, radius){
        for(let lx=0; lx<this.size_l; lx++){
            const offset = this.get_layer_offset(lx);
            for (let x = 0; x < this.size_x; x++) {
                for (let y = 0; y < this.size_y; y++) {
                    const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
                    if (dist < radius) {
                        const falloff = Math.max(0, 1 - (dist / radius) ** 2);
                        this[offset + x * this.size_y + y] = 1;
                    }
                }
            }
        }
    }

	// Scatter multiple random noise squares across the given layer
	seed_random_squares(count, min_size, max_size, layer = 0) {
		for (let i = 0; i < count; i++) {
			const cx = Math.floor(Math.random() * this.size_x);
			const cy = Math.floor(Math.random() * this.size_y);
			const size = min_size + Math.floor(Math.random() * (max_size - min_size));
			this.seed_square(cx, cy, size, layer);
		}
	}
}


class KernelMatrix2D extends Float64Array {
	size = 13; //bounding box size
	half = (this.size-1)/2; //"radius"
	centerpoint = {x: this.half+1, y:this.half+1}

	constructor(size) {
		super(size ** 2)
		this.size = size
		this.half = (size-1)/2
	}

	compute(){throw new Error("Extend this class and add compute()")}
	fmt_statusline(){return `Unknown kernel ${self.size}*${self.size}`}

	iter_elements(fn){
		for (let dx = 0; dx < this.size; dx++) {
			for (let dy = 0; dy < this.size; dy++) {
				fn(dx, dy)
			}
		}
	}
	get(x,y){
		return this[x * this.size + y]
	}
	set(x,y,value){
		return (this[x * this.size + y] = value)
	}
}