You can install `dev` version of pyTorch following this guide.

1) Make sure pyTorch is not installed in your system. Run `python -c import pytorch` and make sure it produces a error. If it is not, remove the existing with either `conda remove pytorch vision torchvision` or `pip uninstall torchvision torch`.

2) Check that you have `cudnn` installed (v6.0 works best). If you have custom cudnn path set environment variables:
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
pip  -vvv install --upgrade --force-reinstall https://github.com/pytorch/torchvision/archive/master.zip
```

3) Check everything is OK with:

```
import torch
print(torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor(1)))
print(torch.backends.cudnn.version())
```
