# PyTorch implementation of iterative winners-take-all

This module is an experimental PyTorch implementation of iWTA. It includes additional metrics that help you to build intuition upon our work. It's also faster, if you have CUDA support.

Here is a snapshot of running the *clustering* experiment.

![PyTorch](../images/pytorch_screenshot.png)

## Quick start

Python 3.6+ is required.

1. Install [PyTorch](https://pytorch.org/)
   ```
   conda install pytorch torchvision cpuonly -c pytorch
   ```
2. Install the requirements
   ```
   pip install git+https://github.com/dizcza/pytorch-mighty.git#egg=pytorch-mighty
   pip install matplotlib
   pip install networkx
   ```

3. Start a [Visdom](https://github.com/facebookresearch/visdom) server by running the following command in a new terminal window
   ```
   python -m visdom.server
   ```

4. Run the experiment on your choice **from the project root directory**
   ```
   python nn/clustering.py
   ```

5. Navigate to http://localhost:8097 to see the training progress as in the picture above.
