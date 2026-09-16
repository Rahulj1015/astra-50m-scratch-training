# Engineering Roadmap

## Completed

- Local Python 3.12 virtual environment
- CUDA-enabled PyTorch installation
- GPU tensor and autograd smoke test
- Explicit linear-regression forward pass, loss, gradients, and update

## Next implementation gates

1. Verify scalar derivatives by hand against autograd.
2. Implement a single neuron without `torch.nn`.
3. Build and test a character tokenizer.
4. Train a tiny next-token predictor on a small local corpus.
5. Implement a decoder-only Transformer from individual components.
6. Add dataset streaming, validation, checkpoints, and measured evaluation.
7. Add inference and sampling only after a checkpoint can be loaded.
8. Add the API and browser client around the tested inference boundary.
9. Add authentication, persistence, file handling, and security hardening.
10. Benchmark scale gates before considering 10M, 30M, 100M, or 300M.

## Honest completion criteria

The project is not complete until each component has executable tests,
documented measurements, a reproducible command, and an explicit limitation
section. A placeholder endpoint or untrained model must not be described as a
working assistant.
