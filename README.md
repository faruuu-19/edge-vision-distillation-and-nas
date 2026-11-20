# Edge Vision Distillation and Architecture Search

This project compares supervised learning and knowledge distillation for three small image classifiers. It uses a ResNet-50 teacher on CIFAR-100 and Optuna to search for a CPU-friendly CNN on CIFAR-10. The deployment code exports trained models to ONNX and measures CPU latency, file size, and peak process memory.

The results folder contains the measured outcomes of two reduced CPU runs. The ImageNet-initialized teacher reached 13.77% CIFAR-100 top-1 accuracy. The selected CIFAR-10 architecture reached 26.17% top-1 accuracy and 1.67 ms p95 latency.

Install a compatible PyTorch and torchvision build for your device, then install the dependencies in requirements.txt. Run the pipeline with python -m edge_lab run-all --config configs/smoke.json for a short integration check or select another profile in configs for full-data training.
