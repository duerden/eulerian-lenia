# Lenia WASM build
# Requires: Emscripten SDK (emcc on PATH)

SRC = lenia.cpp
OUT = lenia.js
EXPORTED = _clear_growth,_apply_rule,_apply_growth,_apply_growth_conserved,_fft_prepare_kernel,_fft_apply_rule,_is_power_of_2,_draw,_alloc_f64,_alloc_u8,_free_f64,_free_u8

CFLAGS = -O2 \
	-s WASM=1 \
	-s EXPORTED_FUNCTIONS='[$(EXPORTED)]' \
	-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPF64","HEAPU8"]' \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s MODULARIZE=1 \
	-s EXPORT_NAME='LeniaModule' \
	--no-entry

all: $(OUT)

$(OUT): $(SRC)
	emcc $(CFLAGS) $(SRC) -o $(OUT)

clean:
	rm -f lenia.js lenia.wasm

.PHONY: all clean
