# adam: custom Adam implementations and experiments
Custom implementations of the Adam family of optimizers and a set of notebooks that compare optimizer behavior on simple models and datasets.

## Summary
- `my_adam.py`: contains three small optimizer classes implemented using plain PyTorch tensor ops:
  - `My_adam`: Adam with bias correction.
  - `My_adamax`: AdaMax variant (infinity-norm second moment).
  - `My_adam_no_bias_correction`: Adam without bias-correction applied.
- `optimize_cnn.ipynb`: trains a CNN on CIFAR-10 and compares optimizers.
- `optimize_logistic_reg.ipynb`: trains a logistic regression model on an IMDB bag-of-words dataset and compares optimizers.
- `vae.ipynb`: trains a small VAE on MNIST and sweeps optimizer hyperparameters.