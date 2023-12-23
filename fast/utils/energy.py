import torch

def get_energy(time, device, verbose=False):
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
                        # Power value in mW, converted to W
                        return float(match.group(1)) / 1000
                    else:
                        return 0.0

                output = subprocess.check_output(["sudo", "powermetrics", "-n", "1", "-i", "1000", "--samplers", "all"], universal_newlines=True,
                                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # Suppress output
                gpu_power = get_power(output, re.compile(r"GPU Power: (\d+) mW"))
                ane_power = get_power(output, re.compile(r"ANE Power: (\d+) mW"))
                power = gpu_power + ane_power
                if verbose: print("Apple power output:", power)

            elif device == torch.device("gpu") or device == torch.device("cuda"):  # NVIDIA GPU
                import pynvml  # included in python 3.9
                
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Returns device handle, index 0 specifies the first GPU
                power = pynvml.nvmlDeviceGetPowerUsage(handle)  # Retrieves power usage for GPU in milliwatts
                power = power / 1000  # convert mW to W
                pynvml.nvmlShutdown()
                if verbose: print("NVIDIA power output:", power)

            elif device == torch.device("cpu"):  # Intel CPU
                # Read energy_uj file
                with open('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', 'r') as file:
                    energy = int(file.read().strip()) / 1_000_000  # Gets energy in microjoules, then converts to joules
                    if verbose: print("Intel energy output:", energy)
                    return energy
                
        except:
            # If we can't get power from the hardware, do a manual estimate
            power = 75.0
            if verbose: print("Error fetching CPU/GPU power -- using default value:", power)

        return power * time