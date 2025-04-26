# import pycuda.driver as cuda
# import pycuda.autoinit
import torch

# def get_cuda_version():
#     cuda_version = cuda.get_version()
#     return f"CUDA version: {cuda_version[0]}.{cuda_version[1]}"
#
# print(get_cuda_version())
# 在parser.add_argument区域添加
# parser.add_argument('--split_ratios', type=float, nargs=3,
#                     default=[0.8, 0.1, 0.1],
#                     help='train/val/test split ratios')
# parser.add_argument('--cls_threshold', type=str, default='median',
#                     choices=['median', 'mean', 'auto'],
#                     help='classification threshold strategy')
# print(torch.cuda.is_available())