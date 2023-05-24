# nvidiaMultiPipe Library

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/your-username/nvidiaMultiPipe/blob/main/LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/your-username/nvidiaMultiPipe)](https://github.com/your-username/nvidiaMultiPipe/issues)
[![GitHub Stars](https://img.shields.io/github/stars/your-username/nvidiaMultiPipe)](https://github.com/your-username/nvidiaMultiPipe/stargazers)

The nvidiaMultiPipe library is a powerful tool that reduces TensorRT inference time by leveraging the multipipe method. It enables efficient parallel processing of multiple input samples using NVIDIA TensorRT, resulting in improved performance for deep learning inference tasks.

![alt text](https://github.com/enzo-gonzalez/multipipe/blob/main/doc/gpu.webp)

## Features

- **Multipipe method**: The library implements the multipipe technique to optimize TensorRT inference time by processing multiple input samples simultaneously.

- **High-performance**: By leveraging NVIDIA TensorRT's powerful optimization capabilities and parallel processing, nvidiaMultiPipe achieves fast and efficient inference for deep learning models.

- **Easy integration**: The library provides a simple and user-friendly API, making it easy to integrate into existing projects and workflows.

- **Compatibility**: nvidiaMultiPipe is compatible with a wide range of NVIDIA GPUs and supports popular deep learning frameworks like TensorFlow and PyTorch.

## Installation

You can install nvidiaMultiPipe by following these steps:

1. Clone the repository:
2. Build the library using the provided build system (CMake, Makefile, etc.). Please refer to the installation instructions in the repository for detailed steps.

3. Include the necessary headers and link against the nvidiaMultiPipe library in your project.

## Usage

To use nvidiaMultiPipe in your project, follow these steps:

1. Initialize the library and set the desired configuration parameters.

2. Load your trained TensorRT model into the library.

3. Prepare the input data for inference.

4. Call the `infer()` function with the prepared input data. The library will utilize the multipipe method to process multiple input samples concurrently and provide inference results.

5. Retrieve the inference results and perform any necessary post-processing.

For detailed usage instructions, API documentation, and examples, please refer to the [Documentation](https://github.com/your-username/nvidiaMultiPipe/wiki) in the repository.

## Contributing

Contributions to nvidiaMultiPipe are welcome! If you encounter any issues, have suggestions, or want to contribute improvements or new features, please open an issue or submit a pull request. For more information, please refer to the [Contributing Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

We would like to thank the contributors and the open-source community for their valuable contributions and support to the nvidiaMultiPipe project.

## Contact

For any questions or inquiries, feel free to reach out to us at [enzo.gonzalez.almeria@gmail.com](mailto:your-email@example.com).


