You can install `dev` version of pyTorch following this guide.

1) Make sure pyTorch is not installed in your system. Run `python -c "import pytorch"` and check if it produces an import error. If it does not, remove the existing installation with either `conda remove pytorch vision torchvision` or `pip uninstall torchvision torch`.

2) Check that you have [`CuDNN`](https://developer.nvidia.com/cudnn) installed (v6.0 works best). If you have custom CuDNN path set environment variables:
```
export CUDNN_LIB_DIR='path/to/cudnn/lib64'
export CUDNN_INCLUDE_DIR='path/to/cudnn/include'
```
And add this line to your `.bashrc` file:
```
LD_LIBRARY_PATH="path/to/cudnn/lib64:$LD_LIBRARY_PATH"
```
3) Install pyTorch with:
```
pip  -vvv install --upgrade --force-reinstall https://github.com/pytorch/pytorch/archive/master.zip
pip  -vvv install --upgrade --force-reinstall https://github.com/pytorch/vision/archive/master.zip
```

3) Check everything is OK with:

```
import torch
print(torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor(1)))
print(torch.backends.cudnn.version())
```
It should print your CuDNN version.
