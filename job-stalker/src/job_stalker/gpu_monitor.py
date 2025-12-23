import os
import re
import subprocess
import sys
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger("gpu_monitor")

IS_LINUX = sys.platform.startswith("linux")


class GPUType(Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    AMD_ROCM = "amd_rocm"
    AMD_SYSFS = "amd_sysfs"
    NONE = None


@dataclass
class GPUInfo:
    available: bool = False
    gpu_type: str = ""
    name: str = ""
    short_name: str = ""
    utilization: float = 0
    memory_utilization: float = 0
    memory_used_mb: float = 0
    memory_total_mb: float = 0
    temperature: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "type": self.gpu_type,
            "name": self.name,
            "short_name": self.short_name,
            "utilization": self.utilization,
            "memory_utilization": self.memory_utilization,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "temperature": self.temperature,
        }


class GPUDetector:
    _instance: Optional["GPUDetector"] = None
    _gpu_type: GPUType = GPUType.NONE
    _gpu_name: Optional[str] = None
    _initialized: bool = False

    def __new__(cls) -> "GPUDetector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._detect_gpu()
            GPUDetector._initialized = True

    def _detect_gpu(self) -> None:
        # 1. Try NVIDIA (pynvml)
        if self._try_nvidia():
            return

        # 2. Try AMD (pyamdgpuinfo)
        if self._try_amd_pyamdgpuinfo():
            return

        # 3. Try AMD (rocm-smi CLI)
        if self._try_amd_rocm():
            return

        # 4. Try AMD (sysfs fallback)
        if self._try_amd_sysfs():
            return

        log.info("No GPU detected")

    def _try_nvidia(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                GPUDetector._gpu_type = GPUType.NVIDIA
                GPUDetector._gpu_name = name
                log.info(f"Detected NVIDIA GPU: {name}")
                return True
        except Exception as e:
            log.debug(f"NVIDIA detection failed: {e}")
        return False

    def _try_amd_pyamdgpuinfo(self) -> bool:
        try:
            import pyamdgpuinfo

            if pyamdgpuinfo.detect_gpus() > 0:
                gpu = pyamdgpuinfo.get_gpu(0)
                name = gpu.name if hasattr(gpu, "name") else "AMD GPU"
                GPUDetector._gpu_type = GPUType.AMD
                GPUDetector._gpu_name = name
                log.info(f"Detected AMD GPU via pyamdgpuinfo: {name}")
                return True
        except Exception as e:
            log.debug(f"AMD pyamdgpuinfo detection failed: {e}")
        return False

    def _try_amd_rocm(self) -> bool:
        # rocm-smi is Linux-only
        if not IS_LINUX:
            return False
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                name = "AMD GPU"
                for line in result.stdout.split("\n"):
                    if "Card series" in line or "GPU" in line:
                        name = line.split(":")[-1].strip() if ":" in line else "AMD GPU"
                        break
                GPUDetector._gpu_type = GPUType.AMD_ROCM
                GPUDetector._gpu_name = name
                log.info(f"Detected AMD GPU via rocm-smi: {name}")
                return True
        except Exception as e:
            log.debug(f"AMD rocm-smi detection failed: {e}")
        return False

    def _try_amd_sysfs(self) -> bool:
        # sysfs is Linux-only
        if not IS_LINUX:
            return False
        try:
            drm_path = "/sys/class/drm"
            if not os.path.exists(drm_path):
                return False

            for card in os.listdir(drm_path):
                if card.startswith("card") and card[4:].isdigit():
                    device_path = os.path.join(drm_path, card, "device")
                    vendor_path = os.path.join(device_path, "vendor")
                    if os.path.exists(vendor_path):
                        with open(vendor_path) as f:
                            vendor = f.read().strip()
                        # AMD vendor ID = 0x1002
                        if vendor == "0x1002":
                            name = "AMD GPU"
                            name_path = os.path.join(device_path, "product_name")
                            if os.path.exists(name_path):
                                with open(name_path) as f:
                                    name = f.read().strip()
                            GPUDetector._gpu_type = GPUType.AMD_SYSFS
                            GPUDetector._gpu_name = name
                            log.info(f"Detected AMD GPU via sysfs: {name}")
                            return True
        except Exception as e:
            log.debug(f"AMD sysfs detection failed: {e}")
        return False

    @property
    def gpu_type(self) -> GPUType:
        return self._gpu_type

    @property
    def gpu_name(self) -> Optional[str]:
        return self._gpu_name

    @property
    def has_gpu(self) -> bool:
        return self._gpu_type != GPUType.NONE

    def _get_short_name(self, name: str, gpu_type: GPUType) -> str:
        if gpu_type == GPUType.NVIDIA:
            match = re.search(r"(RTX|GTX|Quadro)\s*(\d{3,4})\s*(Ti|SUPER)?", name, re.I)
            if match:
                short = match.group(1).upper() + " " + match.group(2)
                if match.group(3):
                    short += " " + match.group(3)
                return short
        else:  # AMD variants
            match = re.search(r"(RX|Radeon)\s*(\d{3,4})\s*(XT|XTX)?", name, re.I)
            if match:
                short = "RX " + match.group(2)
                if match.group(3):
                    short += " " + match.group(3)
                return short
        return name or "GPU"

    def get_info(self) -> Optional[GPUInfo]:
        if self._gpu_type == GPUType.NONE:
            return None

        try:
            if self._gpu_type == GPUType.NVIDIA:
                return self._get_nvidia_info()
            elif self._gpu_type == GPUType.AMD:
                return self._get_amd_pyamdgpuinfo_info()
            elif self._gpu_type == GPUType.AMD_ROCM:
                return self._get_amd_rocm_info()
            elif self._gpu_type == GPUType.AMD_SYSFS:
                return self._get_amd_sysfs_info()
        except Exception as e:
            log.error(f"GPU info error: {e}")
            return None

        return None

    def _get_nvidia_info(self) -> GPUInfo:
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = 0

        return GPUInfo(
            available=True,
            gpu_type="nvidia",
            name=self._gpu_name or "",
            short_name=self._get_short_name(self._gpu_name or "", GPUType.NVIDIA),
            utilization=utilization.gpu,
            memory_utilization=utilization.memory,
            memory_used_mb=round(mem_info.used / (1024 * 1024)),
            memory_total_mb=round(mem_info.total / (1024 * 1024)),
            temperature=temp,
        )

    def _get_amd_pyamdgpuinfo_info(self) -> GPUInfo:
        import pyamdgpuinfo

        gpu = pyamdgpuinfo.get_gpu(0)

        gpu_util = gpu.query_load() * 100 if hasattr(gpu, "query_load") else 0
        vram_used = gpu.query_vram_usage() if hasattr(gpu, "query_vram_usage") else 0
        vram_total = gpu.memory_info.get("vram_size", 0) if hasattr(gpu, "memory_info") else 0
        temp = gpu.query_temperature() if hasattr(gpu, "query_temperature") else 0

        mem_used_mb = vram_used / (1024 * 1024) if vram_used else 0
        mem_total_mb = vram_total / (1024 * 1024) if vram_total else 0
        mem_util = (mem_used_mb / mem_total_mb * 100) if mem_total_mb > 0 else 0

        return GPUInfo(
            available=True,
            gpu_type="amd",
            name=self._gpu_name or "",
            short_name=self._get_short_name(self._gpu_name or "", GPUType.AMD),
            utilization=round(gpu_util),
            memory_utilization=round(mem_util),
            memory_used_mb=round(mem_used_mb),
            memory_total_mb=round(mem_total_mb),
            temperature=round(temp) if temp else 0,
        )

    def _get_amd_rocm_info(self) -> GPUInfo:
        gpu_util: float = 0
        mem_used_mb: float = 0
        mem_total_mb: float = 0
        temp: float = 0

        # GPU utilization
        try:
            result = subprocess.run(
                ["rocm-smi", "--showuse"], capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.split("\n"):
                if "GPU use" in line or "%" in line:
                    match = re.search(r"(\d+)\s*%", line)
                    if match:
                        gpu_util = int(match.group(1))
                        break
        except Exception:
            pass

        # Memory
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.split("\n"):
                if "Used" in line:
                    match = re.search(r"(\d+)", line)
                    if match:
                        mem_used_mb = int(match.group(1)) / (1024 * 1024)
                elif "Total" in line:
                    match = re.search(r"(\d+)", line)
                    if match:
                        mem_total_mb = int(match.group(1)) / (1024 * 1024)
        except Exception:
            pass

        # Temperature
        try:
            result = subprocess.run(
                ["rocm-smi", "--showtemp"], capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.split("\n"):
                if "Temperature" in line or "edge" in line.lower():
                    match = re.search(r"(\d+\.?\d*)", line)
                    if match:
                        temp = float(match.group(1))
                        break
        except Exception:
            pass

        mem_util = (mem_used_mb / mem_total_mb * 100) if mem_total_mb > 0 else 0

        return GPUInfo(
            available=True,
            gpu_type="amd",
            name=self._gpu_name or "",
            short_name=self._get_short_name(self._gpu_name or "", GPUType.AMD_ROCM),
            utilization=gpu_util,
            memory_utilization=round(mem_util),
            memory_used_mb=round(mem_used_mb),
            memory_total_mb=round(mem_total_mb),
            temperature=round(temp),
        )

    def _get_amd_sysfs_info(self) -> GPUInfo:
        gpu_util: float = 0
        mem_used_mb: float = 0
        mem_total_mb: float = 0
        temp: float = 0

        # Temperature via hwmon
        try:
            hwmon_base = "/sys/class/drm/card0/device/hwmon"
            if os.path.exists(hwmon_base):
                hwmon_dir = os.path.join(hwmon_base, os.listdir(hwmon_base)[0])
                temp_file = os.path.join(hwmon_dir, "temp1_input")
                if os.path.exists(temp_file):
                    with open(temp_file) as f:
                        temp = int(f.read().strip()) / 1000  # millidegrees to degrees
        except Exception:
            pass

        # GPU busy percent
        try:
            busy_file = "/sys/class/drm/card0/device/gpu_busy_percent"
            if os.path.exists(busy_file):
                with open(busy_file) as f:
                    gpu_util = int(f.read().strip())
        except Exception:
            pass

        # VRAM
        try:
            vram_used_file = "/sys/class/drm/card0/device/mem_info_vram_used"
            vram_total_file = "/sys/class/drm/card0/device/mem_info_vram_total"
            if os.path.exists(vram_used_file):
                with open(vram_used_file) as f:
                    mem_used_mb = int(f.read().strip()) / (1024 * 1024)
            if os.path.exists(vram_total_file):
                with open(vram_total_file) as f:
                    mem_total_mb = int(f.read().strip()) / (1024 * 1024)
        except Exception:
            pass

        mem_util = (mem_used_mb / mem_total_mb * 100) if mem_total_mb > 0 else 0

        return GPUInfo(
            available=True,
            gpu_type="amd",
            name=self._gpu_name or "",
            short_name=self._get_short_name(self._gpu_name or "", GPUType.AMD_SYSFS),
            utilization=gpu_util,
            memory_utilization=round(mem_util),
            memory_used_mb=round(mem_used_mb),
            memory_total_mb=round(mem_total_mb),
            temperature=round(temp),
        )


# Module-level singleton instance
_detector: Optional[GPUDetector] = None


def get_gpu_detector() -> GPUDetector:
    global _detector
    if _detector is None:
        _detector = GPUDetector()
    return _detector


def get_gpu_info() -> Optional[Dict[str, Any]]:
    detector = get_gpu_detector()
    info = detector.get_info()
    return info.to_dict() if info else None


def has_gpu() -> bool:
    return get_gpu_detector().has_gpu
