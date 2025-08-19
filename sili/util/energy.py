import time
import psutil
import subprocess
import matplotlib.pyplot as plt
from collections import deque

class HzRecord:
    def __init__(self):
        self.t_prev = time.time()
        self.t_now = time.time()

    def update(self):
        self.t_prev = self.t_now
        self.t_now = time.time()

    def get_hz(self):
        return 1.0 / (self.t_now - self.t_prev)

    def hz_dist(self, goal):
        return goal - self.get_hz()

    def dt_diff(self, goal):
        return (self.t_now - self.t_prev) - goal

class HzInfo:
    def __init__(self):
        self.actual_hz = 0.0
        self.hz_dist = 0.0
        self.dt_dist = 0.0

    def update(self, hz_record, goal_hz):
        self.actual_hz = hz_record.get_hz()
        self.hz_dist = hz_record.hz_dist(goal_hz)
        self.dt_dist = hz_record.dt_diff(1.0 / goal_hz)

class SystemMetrics:
    def __init__(self):
        self.hz_record = HzRecord()
        self.cpu_temp = {}
        self.cpu_percent = []
        self.cpu_freq = []
        self.cpu_loadavg = []
        self.cpu_fans = {}
        self.gpu_temps = []
        self.gpu_fans = []
        self.gpu_power = []
        self.gpu_mem = []
        self.gpu_util = []
        self.ram_usage = None
        self.swap_usage = None
        self.hdd_usage = None
        self.hdd_counters = {}
        self.battery_level = None

    def get_cpu_temp(self):
        return psutil.sensors_temperatures()

    def get_gpu_temp(self):
        temps = []
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name,gpu_bus_id,gpu_uuid,temperature.gpu', '--format=csv,noheader'], capture_output=True, text=True, check=True)
            for l in result.stdout.strip().split('\n'):
                q = l.split(',')
                temps.append([q[0].strip(), q[1].strip(), q[2].strip(), float(q[3])])
        except subprocess.CalledProcessError:
            pass
        except Exception as e:
            print(f"NVIDIA query failed: {e}")
        try:
            result = subprocess.run(['rocm-smi', '--showid', '--showtemp', '--csv'], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            headers = lines[0].split(',')
            device_idx = headers.index('card') if 'card' in headers else None
            name_idx = headers.index('GPU Name') if 'GPU Name' in headers else None
            bus_idx = headers.index('Bus ID') if 'Bus ID' in headers else None
            temp_idx = headers.index('Temperature (Sensor edge) (C)') if 'Temperature (Sensor edge) (C)' in headers else None
            uuid_idx = headers.index('Unique ID') if 'Unique ID' in headers else None
            for line in lines[1:]:
                fields = line.split(',')
                valid_indices = [i for i in [device_idx, name_idx, bus_idx, temp_idx, uuid_idx] if i is not None]
                if not valid_indices or len(fields) <= max(valid_indices):
                    continue
                gpu_name = fields[name_idx].strip() if name_idx is not None else f"AMD GPU {fields[device_idx]}" if device_idx is not None else "Unknown AMD GPU"
                bus_id = fields[bus_idx].strip() if bus_idx is not None else ""
                uuid = fields[uuid_idx].strip() if uuid_idx is not None else f"AMD-{bus_id or 'unknown'}"
                try:
                    temp = float(fields[temp_idx]) if temp_idx is not None else 0.0
                except (ValueError, IndexError):
                    temp = 0.0
                temps.append([gpu_name, bus_id, uuid, temp])
        except subprocess.CalledProcessError:
            pass
        except Exception as e:
            print(f"AMD query failed: {e}")
        if not temps:
            raise SystemError("No GPU temperature sensors found")
        return temps

    def get_gpu_power(self):
        results = []
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,pci.bus_id,uuid,power.draw,power.limit', '--format=csv,noheader'], capture_output=True, text=True, check=True)
            for line in result.stdout.strip().split('\n'):
                fields = line.split(',')
                if len(fields) < 5:
                    continue
                gpu_name = fields[0].strip()
                bus_id = fields[1].strip()
                uuid = fields[2].strip()
                try:
                    power_usage = float(fields[3].split()[0])
                    power_cap = float(fields[4].split()[0])
                except (ValueError, IndexError):
                    power_usage, power_cap = 0.0, 0.0
                results.append([gpu_name, bus_id, uuid, [power_usage, power_cap]])
        except subprocess.CalledProcessError:
            pass
        except Exception as e:
            print(f"NVIDIA power query failed: {e}")
        try:
            result = subprocess.run(['rocm-smi', '--showid', '--showpower', '--csv'], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            headers = lines[0].split(',')
            device_idx = headers.index('card') if 'card' in headers else None
            name_idx = headers.index('GPU Name') if 'GPU Name' in headers else None
            bus_idx = headers.index('Bus ID') if 'Bus ID' in headers else None
            power_idx = headers.index('Average Power (W)') if 'Average Power (W)' in headers else None
            uuid_idx = headers.index('Unique ID') if 'Unique ID' in headers else None
            valid_indices = [i for i in [device_idx, name_idx, bus_idx, power_idx, uuid_idx] if i is not None]
            for line in lines[1:]:
                fields = line.split(',')
                if not valid_indices or len(fields) <= max(valid_indices):
                    continue
                gpu_name = fields[name_idx].strip() if name_idx is not None else f"AMD GPU {fields[device_idx]}" if device_idx is not None else "Unknown AMD GPU"
                bus_id = fields[bus_idx].strip() if bus_idx is not None else ""
                uuid = fields[uuid_idx].strip() if uuid_idx is not None else f"AMD-{bus_id or 'unknown'}"
                try:
                    power_usage = float(fields[power_idx]) if power_idx is not None else 0.0
                    power_cap = None
                except (ValueError, IndexError):
                    power_usage, power_cap = 0.0, None
                results.append([gpu_name, bus_id, uuid, [power_usage, power_cap]])
        except subprocess.CalledProcessError:
            pass
        except Exception as e:
            print(f"AMD power query failed: {e}")
        if not results:
            raise SystemError("No GPU power sensors found")
        return results

    def get_gpu_memory(self):
        results = []
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,pci.bus_id,uuid,memory.used,memory.total', '--format=csv,noheader'], capture_output=True, text=True, check=True)
            for line in result.stdout.strip().split('\n'):
                fields = line.split(',')
                if len(fields) < 5:
                    continue
                gpu_name = fields[0].strip()
                bus_id = fields[1].strip()
                uuid = fields[2].strip()
                try:
                    memory_used = float(fields[3].split()[0])
                    memory_total = float(fields[4].split()[0])
                except (ValueError, IndexError):
                    memory_used, memory_total = 0.0, 0.0
                results.append([gpu_name, bus_id, uuid, [memory_used, memory_total]])
        except subprocess.CalledProcessError:
            pass
        except Exception as e:
            print(f"NVIDIA memory query failed: {e}")
        try:
            result = subprocess.run(['rocm-smi', '--showid', '--showmeminfo', 'vram', '--csv'], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            headers = lines[0].split(',')
            device_idx = headers.index('card') if 'card' in headers else None
            name_idx = headers.index('GPU Name') if 'GPU Name' in headers else None
            bus_idx = headers.index('Bus ID') if 'Bus ID' in headers else None
            used_idx = headers.index('VRAM Used (MB)') if 'VRAM Used (MB)' in headers else None
            total_idx = headers.index('VRAM Total (MB)') if 'VRAM Total (MB)' in headers else None
            uuid_idx = headers.index('Unique ID') if 'Unique ID' in headers else None
            valid_indices = [i for i in [device_idx, name_idx, bus_idx, used_idx, total_idx, uuid_idx] if i is not None]
            for line in lines[1:]:
                fields = line.split(',')
                if not valid_indices or len(fields) <= max(valid_indices):
                    continue
                gpu_name = fields[name_idx].strip() if name_idx is not None else f"AMD GPU {fields[device_idx]}" if device_idx is not None else "Unknown AMD GPU"
                bus_id = fields[bus_idx].strip() if bus_idx is not None else ""
                uuid = fields[uuid_idx].strip() if uuid_idx is not None else f"AMD-{bus_id or 'unknown'}"
                try:
                    memory_used = float(fields[used_idx]) if used_idx is not None else 0.0
                    memory_total = float(fields[total_idx]) if total_idx is not None else 0.0
                except (ValueError, IndexError):
                    memory_used, memory_total = 0.0, 0.0
                results.append([gpu_name, bus_id, uuid, [memory_used, memory_total]])
        except subprocess.CalledProcessError:
            pass
        except Exception as e:
            print(f"AMD memory query failed: {e}")
        if not results:
            raise SystemError("No GPU memory sensors found")
        return results

    def get_gpu_utilization(self):
        results = []
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,pci.bus_id,uuid,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True, check=True)
            for line in result.stdout.strip().split('\n'):
                fields = line.split(',')
                if len(fields) < 4:
                    continue
                gpu_name = fields[0].strip()
                bus_id = fields[1].strip()
                uuid = fields[2].strip()
                try:
                    utilization = float(fields[3].split()[0])
                except (ValueError, IndexError):
                    utilization = 0.0
                results.append([gpu_name, bus_id, uuid, utilization])
        except subprocess.CalledProcessError:
            pass
        except Exception as e:
            print(f"NVIDIA utilization query failed: {e}")
        try:
            result = subprocess.run(['rocm-smi', '--showid', '--showutilization', '--csv'], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            headers = lines[0].split(',')
            device_idx = headers.index('card') if 'card' in headers else None
            name_idx = headers.index('GPU Name') if 'GPU Name' in headers else None
            bus_idx = headers.index('Bus ID') if 'Bus ID' in headers else None
            util_idx = headers.index('GPU Activity (%)') if 'GPU Activity (%)' in headers else None
            uuid_idx = headers.index('Unique ID') if 'Unique ID' in headers else None
            valid_indices = [i for i in [device_idx, name_idx, bus_idx, util_idx, uuid_idx] if i is not None]
            for line in lines[1:]:
                fields = line.split(',')
                if not valid_indices or len(fields) <= max(valid_indices):
                    continue
                gpu_name = fields[name_idx].strip() if name_idx is not None else f"AMD GPU {fields[device_idx]}" if device_idx is not None else "Unknown AMD GPU"
                bus_id = fields[bus_idx].strip() if bus_idx is not None else ""
                uuid = fields[uuid_idx].strip() if uuid_idx is not None else f"AMD-{bus_id or 'unknown'}"
                try:
                    utilization = float(fields[util_idx]) if util_idx is not None else 0.0
                except (ValueError, IndexError):
                    utilization = 0.0
                results.append([gpu_name, bus_id, uuid, utilization])
        except subprocess.CalledProcessError:
            pass
        except Exception as e:
            print(f"AMD utilization query failed: {e}")
        if not results:
            raise SystemError("No GPU utilization sensors found")
        return results

    def get_gpu_fan_speed(self):
        results = []
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,pci.bus_id,uuid,fan.speed', '--format=csv,noheader'], capture_output=True, text=True, check=True)
            for line in result.stdout.strip().split('\n'):
                fields = line.split(',')
                if len(fields) < 4:
                    continue
                gpu_name = fields[0].strip()
                bus_id = fields[1].strip()
                uuid = fields[2].strip()
                try:
                    fan_speed = float(fields[3].split()[0]) if fields[3].strip() != 'N/A' else None
                except (ValueError, IndexError):
                    fan_speed = None
                results.append([gpu_name, bus_id, uuid, fan_speed])
        except subprocess.CalledProcessError:
            pass
        except Exception as e:
            print(f"NVIDIA fan speed query failed: {e}")
        try:
            result = subprocess.run(['rocm-smi', '--showid', '--showfan', '--csv'], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            headers = lines[0].split(',')
            device_idx = headers.index('card') if 'card' in headers else None
            name_idx = headers.index('GPU Name') if 'GPU Name' in headers else None
            bus_idx = headers.index('Bus ID') if 'Bus ID' in headers else None
            fan_idx = headers.index('Fan Speed (%)') if 'Fan Speed (%)' in headers else None
            uuid_idx = headers.index('Unique ID') if 'Unique ID' in headers else None
            valid_indices = [i for i in [device_idx, name_idx, bus_idx, fan_idx, uuid_idx] if i is not None]
            for line in lines[1:]:
                fields = line.split(',')
                if not valid_indices or len(fields) <= max(valid_indices):
                    continue
                gpu_name = fields[name_idx].strip() if name_idx is not None else f"AMD GPU {fields[device_idx]}" if device_idx is not None else "Unknown AMD GPU"
                bus_id = fields[bus_idx].strip() if bus_idx is not None else ""
                uuid = fields[uuid_idx].strip() if uuid_idx is not None else f"AMD-{bus_id or 'unknown'}"
                try:
                    fan_speed = float(fields[fan_idx]) if fan_idx is not None else None
                except (ValueError, IndexError):
                    fan_speed = None
                results.append([gpu_name, bus_id, uuid, fan_speed])
        except subprocess.CalledProcessError:
            pass
        except Exception as e:
            print(f"AMD fan speed query failed: {e}")
        if not results:
            raise SystemError("No GPU fan speed sensors found")
        return results

    def get_ram_usage(self):
        return psutil.virtual_memory()

    def get_hdd_space(self):
        return psutil.disk_usage('/')

    def get_battery_level(self):
        battery = psutil.sensors_battery()
        return battery if battery else "No Battery"

    def update_metrics(self):
        psutil.cpu_percent(interval=None, percpu=True)
        #time.sleep(0.5)
        self.cpu_temp = self.get_cpu_temp()
        self.cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
        self.cpu_freq = psutil.cpu_freq(percpu=True)
        self.cpu_loadavg = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
        self.cpu_fans = psutil.sensors_fans()
        self.gpu_temps = self.get_gpu_temp()
        self.gpu_fans = self.get_gpu_fan_speed()
        self.gpu_power = self.get_gpu_power()
        self.gpu_mem = self.get_gpu_memory()
        self.gpu_util = self.get_gpu_utilization()
        self.ram_usage = self.get_ram_usage()
        self.swap_usage = psutil.swap_memory()
        self.hdd_usage = self.get_hdd_space()
        self.hdd_counters = psutil.disk_io_counters(perdisk=True)
        self.battery_level = self.get_battery_level()
        self.hz_record.update()

    def report_hz(self, goal_hz=60.0):
        hz_info = HzInfo()
        hz_info.update(self.hz_record, goal_hz)
        return hz_info

class GPUActivationSparsityManager:
    def __init__(self, gpu_uuid, max_temp=70.0, goal_hz=10.0, min_sparsity = 0.001, max_sparsity=0.5, adj_per_sec = 0.1):
        self.gpu_uuid = gpu_uuid
        self.max_temp = max_temp
        self.goal_hz = goal_hz
        self.max_util = None
        self.max_power = None
        self.sparsity = 0.5
        self.max_sparsity = max_sparsity
        self.min_sparsity = min_sparsity
        self.t_prev = None
        self.adj_per_sec = adj_per_sec

    def set_max_util(self, max_util):
        self.max_util = max_util

    def set_max_power(self, max_power):
        self.max_power = max_power

    def update(self, metrics):
        gpu_data = next((g for g in metrics.gpu_temps if g[2] == self.gpu_uuid), None)
        if self.t_prev is None:
            self.t_prev = time.time()
        t = time.time()
        if not gpu_data:
            return self.sparsity
        temp = gpu_data[3]
        hz_info = metrics.report_hz(self.goal_hz)
        util_data = next((g for g in metrics.gpu_util if g[2] == self.gpu_uuid), [0, 0, 0, 0.0])[3]
        power_data = next((g for g in metrics.gpu_power if g[2] == self.gpu_uuid), [0, 0, 0, [0.0, 0.0]])[3][0]

        adj = (t-self.t_prev)*self.adj_per_sec

        if temp > self.max_temp or hz_info.actual_hz < self.goal_hz or \
           (self.max_util and util_data > self.max_util) or \
           (self.max_power and power_data > self.max_power):
            self.sparsity = max(self.min_sparsity, self.sparsity - adj)
        else:
            self.sparsity = min(self.max_sparsity, self.sparsity + adj)
        self.t_prev = t
        return self.sparsity

class GPUSynapseNeuronManager:
    def __init__(self, gpu_uuid, mem_setpoint=0.8, synapse_size=12, neuron_size=100):
        self.gpu_uuid = gpu_uuid
        self.mem_setpoint = mem_setpoint
        self.synapse_size = synapse_size
        self.neuron_size = neuron_size
        self.max_synapses = 0
        self.max_neurons = 0

    def set_mem_setpoint(self, setpoint):
        self.mem_setpoint = setpoint

    def update(self, metrics):
        gpu_mem = next((g for g in metrics.gpu_mem if g[2] == self.gpu_uuid), [0, 0, 0, [0, 1000]])[3]
        used, total = gpu_mem[0], gpu_mem[1]
        target_used = total * self.mem_setpoint
        available_mb = target_used - used
        self.max_synapses = int(available_mb * 1e6 / self.synapse_size)
        self.max_neurons = int(available_mb * 1e6 / self.neuron_size)
        return self.max_synapses, self.max_neurons

class CPUActivationSparsityManager:
    def __init__(self, max_temp=70.0, goal_hz=10.0, min_sparsity=0.001, max_sparsity=0.5, adj_per_sec = 0.1):
        self.max_temp = max_temp
        self.goal_hz = goal_hz
        self.max_percent = None
        self.max_freq = None
        self.sparsity = 0.5
        self.max_sparsity = max_sparsity
        self.min_sparsity = min_sparsity
        self.adj_per_sec = adj_per_sec
        self.t_prev = None

    def set_max_percent(self, max_percent):
        self.max_percent = max_percent

    def set_max_freq(self, max_freq):
        self.max_freq = max_freq

    def update(self, metrics):
        cpu_temp = max(t.current for t in metrics.cpu_temp.get('coretemp', [])) if metrics.cpu_temp.get('coretemp') else 50.0
        cpu_percent = max(metrics.cpu_percent) if metrics.cpu_percent else 50.0
        cpu_freq = max(f.current for f in metrics.cpu_freq if f) if metrics.cpu_freq else 4000.0
        hz_info = metrics.report_hz(self.goal_hz)

        if self.t_prev is None:
            self.t_prev = time.time()
        t = time.time()
        adj = (t-self.t_prev)*self.adj_per_sec

        if cpu_temp > self.max_temp or hz_info.actual_hz < self.goal_hz or \
           (self.max_percent and cpu_percent > self.max_percent) or \
           (self.max_freq and cpu_freq > self.max_freq):
            self.sparsity = max(self.min_sparsity, self.sparsity - adj)
        else:
            self.sparsity = min(self.max_sparsity, self.sparsity + adj)
        self.t_prev = t
        return self.sparsity

class CPUSynapseNeuronManager:
    def __init__(self, ram_setpoint=0.8, swap_setpoint=0.8, synapse_size=12, neuron_size=12, sips_max = 100_000_000, sops_max = 100_000_000, swap_time_window=5.0, swap_measure_interval=0.1):
        self.ram_setpoint = ram_setpoint
        self.swap_setpoint = swap_setpoint
        self.synapse_size = synapse_size
        self.neuron_size = neuron_size
        self.sips_max = sips_max  # swap in per second
        self.sops_max = sops_max  # swap out per second
        self.max_synapses = 0
        self.max_neurons = 0
        self.prev_time = None
        self.swap_in = []
        self.swap_out = []
        self.swap_stamps = []
        self.swap_time_window = swap_time_window
        self.swap_measure_interval = swap_measure_interval

    def set_ram_setpoint(self, setpoint):
        self.ram_setpoint = setpoint

    def update(self, metrics):
        ram = metrics.ram_usage
        swap = metrics.swap_usage
        t = time.time()
        if (self.prev_time is None or t - self.prev_time>self.swap_measure_interval) and self.swap_setpoint!=0:
            self.swap_in.append(swap.sin)
            self.swap_out.append(swap.sout)
            self.swap_stamps.append(time.time())
        while len(self.swap_stamps)>0:
            if t-self.swap_stamps[0]:
                self.swap_in.pop(0)
                self.swap_out.pop(0)
                self.swap_stamps.pop(0)
            else:
                break
        if len(self.swap_stamps)>1:
            sips = (self.swap_in[-1] - self.swap_in[0])/(self.swap_stamps[-1] - self.swap_stamps[0])
            sops = (self.swap_out[-1] - self.swap_out[0])/(self.swap_stamps[-1] - self.swap_stamps[0])
        else:
            sips = 0
            sops = 0

        target_used = ram.total * self.ram_setpoint / 1e6
        available_mb = target_used - (ram.used / 1e6)
        target_used_swap = swap.total * self.swap_setpoint / 1e6
        available_swap = target_used_swap - (swap.used / 1e6)

        if sips>self.sips_max or sops>self.sops_max:
            self.max_synapses = int((available_mb) * 1e6 / self.synapse_size)
            self.max_neurons = int((available_mb) * 1e6 / self.neuron_size)
        else:
            self.max_synapses = int((available_mb+available_swap) * 1e6 / self.synapse_size)
            self.max_neurons = int((available_mb+available_swap) * 1e6 / self.neuron_size)
        self.prev_time = t
        return self.max_synapses, self.max_neurons, self.sips_max-sips, self.sops_max-sops

class HDDActivationSparsityManager:
    def __init__(self, disk_names, goal_hz=10.0, max_io_rate=1e6, min_sparsity=0.001, max_sparsity=0.5, adj_per_sec=0.1):
        self.disk_names = disk_names
        self.goal_hz = goal_hz
        self.max_io_rate = max_io_rate
        self.sparsity = 0.5
        self.prev_io = {d: {'read': 0, 'write': 0, 'time': time.time()} for d in disk_names}
        self.t_prev = None
        self.min_sparsity = min_sparsity
        self.max_sparsity=max_sparsity
        self.adj_per_sec = adj_per_sec

    def update(self, metrics):
        hz_info = metrics.report_hz(self.goal_hz)
        io_rate = 0.0
        for disk in self.disk_names:
            counters = metrics.hdd_counters.get(disk, None)
            prev = self.prev_io.get(disk, {'read': 0, 'write': 0, 'time': time.time()})
            if counters:
                t_now = time.time()
                dt = t_now - prev['time']
                if dt > 0:
                    io_rate += ((counters.read_bytes - prev['read']) + (counters.write_bytes - prev['write'])) / dt
                self.prev_io[disk] = {'read': counters.read_bytes, 'write': counters.write_bytes, 'time': t_now}

        if self.t_prev is None:
            self.t_prev = time.time()
        t = time.time()
        adj = (t-self.t_prev)*self.adj_per_sec

        if io_rate > self.max_io_rate or hz_info.actual_hz < self.goal_hz:
            self.sparsity = max(self.min_sparsity, self.sparsity - adj)
        else:
            self.sparsity = min(self.max_sparsity, self.sparsity + adj)
        self.t_prev = t
        return self.sparsity

class HDDSynapseNeuronManager:
    def __init__(self, disk_names, space_setpoint=0.8, synapse_size=12, neuron_size=12):
        self.disk_names = disk_names
        self.space_setpoint = space_setpoint
        self.synapse_size = synapse_size
        self.neuron_size = neuron_size
        self.max_synapses = 0
        self.max_neurons = 0

    def set_space_setpoint(self, setpoint):
        self.space_setpoint = setpoint

    def update(self, metrics):
        total, used = 0, 0
        for disk in self.disk_names:
            disk_data = metrics.hdd_usage if disk == '/' else psutil.disk_usage(f'/dev/{disk}')
            total += disk_data.total / 1e6
            used += disk_data.used / 1e6
        target_used = total * self.space_setpoint
        available_mb = target_used - used

        self.max_synapses = int(available_mb * 1e6 / self.synapse_size)
        self.max_neurons = int(available_mb * 1e6 / self.neuron_size)
        return self.max_synapses, self.max_neurons

if __name__ == "__main__":
    metrics = SystemMetrics()
    metrics.update_metrics()  # Initial metrics to get GPU UUIDs
    gpu_managers = []
    gpu_synapse_managers = []
    for gpu in metrics.gpu_temps:
        uuid = gpu[2]
        gpu_managers.append(GPUActivationSparsityManager(uuid, goal_hz=0.5))
        gpu_synapse_managers.append(GPUSynapseNeuronManager(uuid))
    cpu_manager = CPUActivationSparsityManager(goal_hz=0.5)
    cpu_synapse_manager = CPUSynapseNeuronManager()
    hdd_manager = HDDActivationSparsityManager(['sdc3'], goal_hz=0.5)
    hdd_synapse_manager = HDDSynapseNeuronManager(['sdc3'])

    max_points = 100
    times = deque(maxlen=max_points)
    hz_data = deque(maxlen=max_points)

    gpu_sparsity_data = [deque(maxlen=max_points) for _ in gpu_managers]
    gpu_synapse_data = [deque(maxlen=max_points) for _ in gpu_synapse_managers]
    cpu_sparsity_data = deque(maxlen=max_points)
    cpu_synapse_data = deque(maxlen=max_points)
    hdd_sparsity_data = deque(maxlen=max_points)
    hdd_synapse_data = deque(maxlen=max_points)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    start_time = time.time()

    while True:
        #time.sleep(0.01)
        metrics.update_metrics()
        hz_info = metrics.report_hz()
        current_time = time.time() - start_time
        times.append(current_time)
        hz_data.append(hz_info.actual_hz)

        for i, manager in enumerate(gpu_managers):
            gpu_sparsity_data[i].append(manager.update(metrics))
        for i, manager in enumerate(gpu_synapse_managers):
            synapses, _ = manager.update(metrics)
            gpu_synapse_data[i].append(synapses / 1e6)  # Scale to millions
        cpu_sparsity_data.append(cpu_manager.update(metrics))
        cpu_synapse_data.append(cpu_synapse_manager.update(metrics)[0] / 1e6)
        hdd_sparsity_data.append(hdd_manager.update(metrics))
        hdd_synapse_data.append(hdd_synapse_manager.update(metrics)[0] / 1e6)

        ax1.clear()
        ax1.plot(times, hz_data, label='Hz', color='black')
        for i, sparsity in enumerate(gpu_sparsity_data):
            ax1.plot(times, sparsity, label=f'GPU{i} Sparsity', color=f'C{i}')
        ax1.plot(times, cpu_sparsity_data, label='CPU Sparsity', color='blue')
        ax1.plot(times, hdd_sparsity_data, label='HDD Sparsity', color='green')
        ax1.set_title('Sparsity and Hz over Time')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Sparsity (0-1) / Hz')
        ax1.legend()
        ax1.grid(True)

        ax2.clear()
        for i, synapses in enumerate(gpu_synapse_data):
            ax2.plot(times, synapses, label=f'GPU{i} Synapses', color=f'C{i}')
        ax2.plot(times, cpu_synapse_data, label='CPU Synapses', color='blue')
        ax2.plot(times, hdd_synapse_data, label='HDD Synapses', color='green')
        ax2.set_title('Synapses Available')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Synapses Available (Millions)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.pause(0.001)
