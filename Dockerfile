FROM nginx:alpine

COPY index.html index.css index.js lenia.js lenia.wasm matrix.js kernels.js /usr/share/nginx/html/

EXPOSE 80
