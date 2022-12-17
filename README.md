# PINN for wave equation

Using PINN to solve the wave equation by boundary and initial conditions.

See also: [Physics-informed Neural Networks (PINNs) for Wave Propagation and Full Waveform Inversions](https://arxiv.org/abs/2108.12035)

The different descriptions refers to the different loss functions and the different models.

For futher instructions, plase see the demo ([jupyter notebook](./demo/demo.ipynb))

## Example
For scripts please see the files in the directory named `archive`. For descriptions to the model of each script, see [description](archive/description.md)
- `plane-3d-transmission.py` ![3d plane wave](example/3d-transmission_3x20_5000_0.0/figures/Predict3D.gif)
- `gaussian-2d-transmission.py` ![2d gaussian](example/2d-transmission_4x20_5000_0.0/figures/Predict_animated.gif)
- `gaussian-1d.py` (Note that the boundary condition is different from those given in this document. Please see the [demo](./demo/demo.ipynb) for details) ![1d gaussian](example/1d-reflection_4x20_10000_0.001/figures/Predict_animated.gif)

## Contact me
zhengcf@mail.bnu.edu.cn
