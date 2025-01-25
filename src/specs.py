import platform
import os
import sys
import psutil
import torch

def get_system_info() -> dict:
	info = {
		'os': platform.system(),
		'os_version': platform.mac_ver()[0] if platform.system() == 'Darwin' else None,
		'python_version': sys.version.split()[0],
		'ram': f'{round(psutil.virtual_memory().total / 1e9,2)} GB'
	}
	
	return info

def get_pytorch_info() -> dict:
	info = {
		'pytorch_version': torch.__version__,
		'cuda_available': torch.cuda.is_available(),
		'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
		'mps_available': torch.backends.mps.is_available() if platform.system() == 'Darwin' else None,
		'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
		'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None,
		'compute_capabilities': [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None,
	}
	return info

def get_cpu_data() -> dict:
	info = {
		'is_available': True,
		# basic info
		'cpu_model': platform.processor(),
		'architecture': platform.machine(),
		'physical_cores': psutil.cpu_count(logical=False),
		'logical_processors': os.cpu_count(),
		'cpu_frequency': 'coming soon!',
		# pytorch specific
		'supports_avx2': torch._C._cpu._is_avx2_supported(),
		'supports_avx512': torch._C._cpu._is_avx512_supported(),
		'supports_avx512_bf16': torch._C._cpu._is_avx512_bf16_supported(),
		'supports_vnni': torch._C._cpu._is_avx512_vnni_supported(),
		'supports_amx_tile': torch._C._cpu._is_amx_tile_supported(),
		'amx_initialized': torch._C._cpu._init_amx(),
	}
	return info
system_info = get_system_info()
cpu_info = get_cpu_data()
pytorch_info = get_pytorch_info()

print('System Information\n')
for key, value in system_info.items():
	print(f'{key}: {value}')

print('\nCPU Information')
for key, value in cpu_info.items():
	print(f'{key}: {value}')

print('\nPytorch Information')
for key, value in pytorch_info.items():
	print(f'{key}: {value}')
