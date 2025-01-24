import platform
import os
import sys
import psutil
import torch

def get_system_info():
	info = {
		'os': platform.system(),
		'os_version': platform.mac_ver()[0] if platform.system() == 'Darwin' else None,
		'architecture': platform.machine(),
		'python_version': sys.version.split()[0],
		'cpu': platform.processor(),
		'ram': f'{round(psutil.virtual_memory().total / 1e9,2)} GB'
	}
	
	return info

def get_pytorch_info():
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

system_info = get_system_info()
pytorch_info = get_pytorch_info()
print('System Information\n')
for key, value in system_info.items():
	print(f'{key}: {value}')

print('\nPytorch Information')
for key, value in pytorch_info.items():
	print(f'{key}: {value}')
