import torch
def get_energy(time, device):
        """
        Finds energy usage of the given epoch
        Estimates power output of hardware (watts) and computes E = P * delta t
        Returns energy in joules
        """
        try:
            # Try to get power output info from hardware
            if device == torch.device("mps"):  # Apple GPU
                import subprocess
                import re

                def get_power(output, pattern):
                    match = pattern.search(output)
                    if match:
                        # Power value, convert mW to W
                        return float(match.group(1)) / 1000
                    else:
                        return 0.0

                output = subprocess.check_output(["sudo", "powermetrics", "-n", "1", "-i", "1000", "--samplers", "all"], universal_newlines=True,
                                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # Suppress output
                gpu_power = get_power(
                    output, re.compile(r"GPU Power: (\d+) mW"))
                ane_power = get_power(
                    output, re.compile(r"ANE Power: (\d+) mW"))
                power = gpu_power + ane_power

            elif device == torch.device("gpu"):  # NVIDIA GPU
                import pynvml  # incl in python 3.9

                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(
                        0)  # Index 0 represents the first GPU
                    power = pynvml.nvmlDeviceGetPowerUsage(
                        handle)  # Power usage in milliwatts
                    power = power / 1000  # convert mW to W

            elif device == torch.device("cpu"):  # Intel CPU
                # Read energy_uj file
                with open('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', 'r') as file:
                    energy_microjoules = int(file.read().strip())
                    return energy_microjoules / 1_000_000  # convert to joules

        except:
            # If we can't get power from the hardware, do a manual estimate
            power = 75.0

        return power * time