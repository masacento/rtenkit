# rtenkit

A lightweight, high-performance library for machine learning inference by combining [RTen](https://github.com/robertknight/rten) (Rust Tensor engine) with [kitoken](https://github.com/Systemcluster/kitoken) tokenizers. Can be used as a C library.

## Features

* **Efficient Model Inference**: Run machine learning models converted from ONNX format using RTen's optimized runtime
* **Fast Tokenization**: Process text for LLMs with kitoken's high-performance tokenizers (supports BPE, Unigram, WordPiece)
* **Cross-Platform**: Works on Windows, Linux, macOS, and WebAssembly.

## Supported Task Types

rtenkit currently provides built-in support for:

* **Embeddings**: Generate vector representations of text for semantic search and similarity applications

## License

MIT

## Acknowledgements

This project builds upon:
* [RTen](https://github.com/robertknight/rten) by Robert Knight
* [kitoken](https://github.com/Systemcluster/kitoken) by Christian Sdunek