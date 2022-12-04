import re
import platform
import psutil
import cpuinfo

import GPUtil

def get_cpu_features():
    """Get relevant predictors about the CPU"""
    # TODO : Select as many predictors as possible and eliminate the non relevant ones using for eg a lasso path
    cpu_infos = cpuinfo.get_cpu_info()
    return {"CPU_count": cpu_infos['count'],
            "CPU_vendor_id": cpu_infos['vendor_id_raw'],
            "CPU_GHz": float(re.findall("\d+\.\d+", cpu_infos['hz_advertised_friendly'])[0]),
            "core_architecture" : cpu_infos['arch_string_raw']
            }

def get_system_features():
    """Get relevant predictors about the system"""
    # TODO : Add more relevant features.
    return {
        "os" : platform.system()
    }
        

def get_GPU_features():
    """If GPU is available, returns relevant GPU features used to create a energy consumption DataFrame"""
    # TODO: Add more relevant features
    if len(GPUtil.getGPUs()) != 0:
        return {
            "GPU_name" : GPUtil.getGPUs()[0].name
        }

def get_memory_features():
    """Get relevant predictors about the memory"""
    # TODO: Add more relevant features
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "memory_available_B" : mem.available,
        "swap_free_B" : swap.free
    }