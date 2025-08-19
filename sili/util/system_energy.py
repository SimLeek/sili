import time
import psutil
import asyncio
import threading
import logging
from dataclasses import dataclass
from collections import deque
from PySide2.QtWidgets import QApplication
from PySide2.QtCore import QTimer
import pyqtgraph as pg
import sys
import numpy as np
import glob
import os
from filterpy.kalman import KalmanFilter
from filterpy.discrete_bayes import normalize, update, predict

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# cpu sparsity - cpu temperature (70c, raise to this if under) > RAM temp if it exists > hdd temperature if it exists and if swap is used (40c, raise if under) > full power times battery remaining below 50% (min(1,batt/.5)*cpu_power if batt exists else pass) > Hz
# gpu sparsity - gpu temperature (70c, raise to this if under) & GPU memory temperature (70c as well) > full watts times battery remaining below 50% > Hz
# hdd sparsity - hdd temperature (40c, raise if under) > hdd read+write added together for all hard drives > full watts times battery remaining below 50% > Hz
#   update hdd temp every second. It doesn't change that fast
#   note: `sudo modprobe drivetemp` to get hdd temps with just `sensors`. Requires setup
# cpu synapses - HDD swap (if used) lowest write+read HDD with swap, combined divided by current sparsity times max sparsity

@dataclass
class SystemMetricsData:
    """Holds system metrics data collected by ThreadedSystemMetrics."""
    cpu_temp: list  # List of [zone, type, temp] for thermal_zone* CPU temps
    cpu_percent: list
    cpu_freq: list
    cpu_loadavg: list
    cpu_fans: dict
    gpu_info: list  # List of [name, bus_id, uuid, temp, [power_usage, power_cap], [mem_used, mem_total], utilization, fan_speed]
    ram_usage: object
    swap_usage: object
    hdd_usage: object
    hdd_temp: list # List of [name, temp] for each drive
    hdd_counters: dict
    battery_level: object


@dataclass
class FilteredMetricsData:
    """Holds filtered system metrics data."""
    cpu_temp: list  # List of [zone, type, filtered_temp] for thermal_zone* CPU temps
    gpu_info: list  # List of [name, bus_id, uuid, filtered_temp, ...]


class KalmanFilterWrapper:
    """Manages a 1D Kalman filter for a single sensor with Bayesian parameter estimation."""

    def __init__(self, sensor_id: str, initial_value: float = 0.0):
        """Initialize Kalman filter for a sensor.

        Args:
            sensor_id: Unique identifier for the sensor (e.g., thermal zone or GPU UUID).
            initial_value: Initial temperature value.
        """
        self.sensor_id = sensor_id
        self.kf = KalmanFilter(dim_x=1, dim_z=1)
        self.kf.x = np.array([initial_value])  # Initial state (temperature)
        self.kf.F = np.array([[1.]])  # State transition matrix (linear)
        self.kf.H = np.array([[1.]])  # Measurement matrix
        self.kf.P = np.array([[1000.]])  # Initial error covariance
        self.kf.R = 100.0  # Initial measurement noise
        self.kf.Q = .005  # Initial process noise
        self.data_window = deque(maxlen=60)  # 1s of data at 60 FPS
        self.residuals = deque(maxlen=60)  # Store residuals for parameter estimation

    def update(self, measurement: float):
        """Update the Kalman filter with a new measurement and estimate parameters.

        Args:
            measurement: New temperature measurement.
        """
        self.data_window.append(measurement)
        self.kf.predict()
        self.kf.update(measurement)
        self.residuals.append(measurement - self.kf.x[0])

        #return

        # Bayesian parameter estimation (R and Q) every 30 samples (1s)
        if len(self.data_window) > 1: #== self.data_window.maxlen:
            # Estimate measurement noise (R) from variance of residuals
            if len(self.residuals) > 1:
                #print("R",np.var(list(self.residuals)))
                r_var = np.var(list(self.residuals)) if np.var(list(self.residuals)) > 0 else 1.0
                self.kf.R = r_var
                #print("R", r_var)

            # Estimate process noise (Q) based on data variance
            #data_var = np.var(list(self.data_window)) if np.var(list(self.data_window)) > 0 else 0.01
            #print("Q", data_var,)
            #self.kf.Q = data_var# * .01
            # Update P based on steady-state assumption
            #self.kf.P = np.array([[(self.kf.R + self.kf.Q)]])
            #self.kf.P = np.array([[self.kf.R + self.kf.Q]])

    def get_filtered_value(self) -> float:
        """Return the filtered temperature.

        Returns:
            float: Current filtered temperature.
        """
        return self.kf.x[0]


class ThreadedSystemMetrics:
    """Manages continuous, thread-safe collection of system metrics in the background.

    Metrics are updated in a separate thread and accessed via a dataclass for thread-safe retrieval.
    """

    def __init__(self, update_interval: float = 1.0 / 30, use_psutil_cpu_temp: bool = False):
        """Initialize the metrics collector with specified parameters.

        Args:
            update_interval: Time between metric updates in seconds (default: 1/30 ≈ 0.0333).
            use_psutil_cpu_temp: If True, use psutil.sensors_temperatures() for CPU temps (slower, per-core).
                                 If False, use /sys/class/thermal/thermal_zone* (faster, package-level).
        """
        self.update_interval = update_interval
        self.use_psutil_cpu_temp = use_psutil_cpu_temp
        self.lock = threading.Lock()
        self.running = False
        self.update_thread = None
        self.cpu_filters = {}  # Dict of zone -> KalmanFilterWrapper
        self.gpu_filters = {}  # Dict of uuid -> KalmanFilterWrapper

        # Initialize metrics
        with self.lock:
            self.cpu_temp = []
            self.cpu_percent = []
            self.cpu_freq = []
            self.cpu_loadavg = []
            self.cpu_fans = {}
            self.gpu_info = []
            self.ram_usage = None
            self.swap_usage = None
            self.hdd_usage = None
            self.hdd_counters = {}
            self.battery_level = None
            self.filtered_cpu_temp = []
            self.filtered_gpu_info = []

    async def get_gpu_info(self) -> list:
        """Retrieve all GPU metrics (temperature, power, memory, utilization, fan speed) for NVIDIA and AMD GPUs.

        Returns:
            list: List of [name, bus_id, uuid, temp, [power_usage, power_cap], [mem_used, mem_total], utilization, fan_speed]
                  for all detected GPUs, or default list on failure.
        """
        results = []
        # NVIDIA query
        try:
            proc = await asyncio.create_subprocess_exec(
                'nvidia-smi',
                '--query-gpu=name,pci.bus_id,uuid,temperature.gpu,temperature.memory,power.draw,power.limit,memory.used,memory.total,utilization.gpu,fan.speed',
                '--format=csv,noheader',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                for line in stdout.decode().strip().split('\n'):
                    fields = line.split(',')
                    if len(fields) < 10:
                        continue
                    gpu_name = fields[0].strip()
                    bus_id = fields[1].strip()
                    uuid = fields[2].strip()
                    try:
                        temp = float(fields[3])
                        if 'N/A' not in fields[4]:
                            mem_temp = float(fields[4])
                        else:
                            mem_temp = None
                        power_usage = float(fields[5].split()[0])
                        power_cap = float(fields[6].split()[0])
                        mem_used = float(fields[7].split()[0])
                        mem_total = float(fields[8].split()[0])
                        utilization = float(fields[9].split()[0])
                        if 'N/A' not in fields[10]:
                            fan_speed = float(fields[10].split()[0])
                        else:
                            fan_speed = None
                    except (ValueError, IndexError):
                        temp, mem_temp, power_usage, power_cap, mem_used, mem_total, utilization, fan_speed = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None
                    results.append(
                        [gpu_name, bus_id, uuid, temp, mem_temp, [power_usage, power_cap], [mem_used, mem_total], utilization,
                         fan_speed])
                logging.info(f"NVIDIA GPUs detected: {len([r for r in results if 'NVIDIA' in r[0]])}")
        except Exception:
            pass  # Silently ignore NVIDIA query failures

        # AMD query
        try:
            proc = await asyncio.create_subprocess_exec(
                'rocm-smi', '--showid', '--showtemp', '--showpower', '--showmeminfo', 'vram', '--showutilization',
                '--showfan', '--csv',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                lines = stdout.decode().strip().split('\n')
                if len(lines) > 1:
                    headers = lines[0].split(',')
                    device_idx = headers.index('card') if 'card' in headers else None
                    name_idx = headers.index('GPU Name') if 'GPU Name' in headers else None
                    bus_idx = headers.index('Bus ID') if 'Bus ID' in headers else None
                    temp_idx = headers.index(
                        'Temperature (Sensor edge) (C)') if 'Temperature (Sensor edge) (C)' in headers else None
                    power_idx = headers.index('Average Power (W)') if 'Average Power (W)' in headers else None
                    mem_used_idx = headers.index('VRAM Used (MB)') if 'VRAM Used (MB)' in headers else None
                    mem_total_idx = headers.index('VRAM Total (MB)') if 'VRAM Total (MB)' in headers else None
                    util_idx = headers.index('GPU Activity (%)') if 'GPU Activity (%)' in headers else None
                    fan_idx = headers.index('Fan Speed (%)') if 'Fan Speed (%)' in headers else None
                    uuid_idx = headers.index('Unique ID') if 'Unique ID' in headers else None
                    valid_indices = [i for i in
                                     [device_idx, name_idx, bus_idx, temp_idx, power_idx, mem_used_idx, mem_total_idx,
                                      util_idx, fan_idx, uuid_idx] if i is not None]
                    for line in lines[1:]:
                        fields = line.split(',')
                        if not valid_indices or len(fields) <= max(valid_indices, default=-1):
                            continue
                        gpu_name = fields[
                            name_idx].strip() if name_idx is not None else f"AMD GPU {fields[device_idx]}" if device_idx is not None else "Unknown AMD GPU"
                        bus_id = fields[bus_idx].strip() if bus_idx is not None else ""
                        uuid = fields[uuid_idx].strip() if uuid_idx is not None else f"AMD-{bus_id or 'unknown'}"
                        try:
                            temp = float(fields[temp_idx]) if temp_idx is not None else 0.0
                            power_usage = float(fields[power_idx]) if power_idx is not None else 0.0
                            power_cap = None
                            mem_used = float(fields[mem_used_idx]) if mem_used_idx is not None else 0.0
                            mem_total = float(fields[mem_total_idx]) if mem_total_idx is not None else 0.0
                            utilization = float(fields[util_idx]) if util_idx is not None else 0.0
                            fan_speed = float(fields[fan_idx]) if fan_idx is not None else None
                        except (ValueError, IndexError):
                            temp, power_usage, mem_used, mem_total, utilization, fan_speed = 0.0, 0.0, 0.0, 0.0, 0.0, None
                        results.append(
                            [gpu_name, bus_id, uuid, temp, [power_usage, power_cap], [mem_used, mem_total], utilization,
                             fan_speed])
                    logging.info(f"AMD GPUs detected: {len([r for r in results if 'AMD' in r[0]])}")
        except Exception:
            pass  # Silently ignore AMD query failures

        return results if results else [[None, None, None, 0.0, [0.0, None], [0.0, 0.0], 0.0, None]]

    def get_cpu_temp_from_thermal_zones(self) -> list:
        """Read CPU temperatures from /sys/class/thermal/thermal_zone* where type contains 'x86'.

        Returns:
            list: List of [zone, type, temp] tuples for CPU thermal zones, or empty list on failure.
        """
        cpu_temps = []
        try:
            # Find all thermal zones
            thermal_zones = glob.glob('/sys/class/thermal/thermal_zone*')
            for zone in thermal_zones:
                type_file = os.path.join(zone, 'type')
                temp_file = os.path.join(zone, 'temp')
                if os.path.exists(type_file) and os.path.exists(temp_file):
                    with open(type_file, 'r') as f:
                        zone_type = f.read().strip()
                    if 'x86' in zone_type.lower():  # e.g., x86_pkg_temp
                        with open(temp_file, 'r') as f:
                            temp = float(f.read().strip()) / 1000.0  # Convert millidegrees to degrees Celsius
                        cpu_temps.append([os.path.basename(zone), zone_type, temp])
                        logging.info(
                            f"CPU Temp: {temp:.1f} °C, Zone: {zone}, Type: {zone_type}, Timestamp: {time.time():.3f}")
        except Exception as e:
            logging.info(f"Failed to read thermal zone temperatures: {e}")
        return cpu_temps

    def update_metrics(self):
        """Update all system metrics in a thread-safe manner."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with self.lock:
                # Update CPU temperatures
                if self.use_psutil_cpu_temp:
                    self.cpu_temp = psutil.sensors_temperatures().get('coretemp', [])
                    self.filtered_cpu_temp = []
                    for temp in self.cpu_temp:
                        zone = temp.label or f"core_{temp.current}"
                        if zone not in self.cpu_filters:
                            self.cpu_filters[zone] = KalmanFilterWrapper(zone, temp.current)
                        self.cpu_filters[zone].update(temp.current)
                        self.filtered_cpu_temp.append([zone, temp.label, self.cpu_filters[zone].get_filtered_value()])
                else:
                    self.cpu_temp = self.get_cpu_temp_from_thermal_zones()
                    self.filtered_cpu_temp = []
                    for temp in self.cpu_temp:
                        zone = temp[0]  # e.g., thermal_zone0
                        if zone not in self.cpu_filters:
                            self.cpu_filters[zone] = KalmanFilterWrapper(zone, temp[2])
                        self.cpu_filters[zone].update(temp[2])
                        self.filtered_cpu_temp.append([zone, temp[1], self.cpu_filters[zone].get_filtered_value()])

                # Update GPU info
                self.gpu_info = loop.run_until_complete(self.get_gpu_info())
                self.filtered_gpu_info = []
                for gpu in self.gpu_info:
                    uuid = gpu[2]
                    if uuid not in self.gpu_filters:
                        self.gpu_filters[uuid] = KalmanFilterWrapper(uuid, gpu[3])
                    self.gpu_filters[uuid].update(gpu[3])
                    filtered_gpu = gpu.copy()
                    filtered_gpu[3] = self.gpu_filters[uuid].get_filtered_value()
                    self.filtered_gpu_info.append(filtered_gpu)

                # Update other metrics
                self.cpu_percent = psutil.cpu_percent(interval=None, percpu=True) or []
                self.cpu_freq = psutil.cpu_freq(percpu=True) or []
                self.cpu_loadavg = [x / psutil.cpu_count() * 100 for x in
                                    psutil.getloadavg()] if psutil.getloadavg() else []
                self.cpu_fans = psutil.sensors_fans() or {}
                self.ram_usage = psutil.virtual_memory()
                self.swap_usage = psutil.swap_memory() or None
                self.hdd_usage = psutil.disk_usage('/')
                self.hdd_counters = psutil.disk_io_counters(perdisk=True) or {}
                self.battery_level = psutil.sensors_battery() or "No Battery"
        finally:
            loop.close()

    def start(self):
        """Start the background thread for continuous metric updates."""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            logging.info("Started metrics update thread")

    def stop(self):
        """Stop the background thread and clean up resources."""
        if self.running:
            self.running = False
            if self.update_thread:
                self.update_thread.join()
            logging.info("Stopped metrics update thread")

    def _update_loop(self):
        """Internal loop to continuously update metrics while running."""
        while self.running:
            try:
                self.update_metrics()
            except Exception as e:
                logging.info(f"Error updating metrics: {e}")
            time.sleep(self.update_interval)

    def get_metrics(self) -> SystemMetricsData:
        """Retrieve the latest raw system metrics in a thread-safe manner.

        Returns:
            SystemMetricsData: Dataclass containing raw system metrics.
        """
        with self.lock:
            return SystemMetricsData(
                cpu_temp=self.cpu_temp,
                cpu_percent=self.cpu_percent,
                cpu_freq=self.cpu_freq,
                cpu_loadavg=self.cpu_loadavg,
                cpu_fans=self.cpu_fans,
                gpu_info=self.gpu_info,
                ram_usage=self.ram_usage,
                swap_usage=self.swap_usage,
                hdd_usage=self.hdd_usage,
                hdd_counters=self.hdd_counters,
                battery_level=self.battery_level
            )

    def get_filtered_metrics(self) -> FilteredMetricsData:
        """Retrieve the latest filtered system metrics in a thread-safe manner.

        Returns:
            FilteredMetricsData: Dataclass containing Kalman-filtered metrics.
        """
        with self.lock:
            return FilteredMetricsData(
                cpu_temp=self.filtered_cpu_temp,
                gpu_info=self.filtered_gpu_info
            )


if __name__ == "__main__":
    """Example usage: Plot filtered CPU and GPU temperatures over time using pyqtgraph with PySide2 at ~30 FPS."""

    # Initialize QApplication
    app = QApplication(sys.argv)

    # Set up metrics collector
    metrics = ThreadedSystemMetrics(update_interval=1.0 / 60, use_psutil_cpu_temp=False)
    metrics.start()
    time.sleep(0.1)
    current_metrics = metrics.get_filtered_metrics()  # get first info

    # Initialize data storage
    max_points = 900  # 30 FPS * 30 seconds
    times = deque(maxlen=max_points)
    cpu_temp_data = deque(maxlen=max_points)
    gpu_temp_data = [deque(maxlen=max_points) for _ in current_metrics.gpu_info]
    start_time = time.time()
    last_update = start_time

    # Set up pyqtgraph plot with white background
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    win = pg.GraphicsLayoutWidget(show=True, title="CPU and GPU Temperatures")
    win.resize(800, 400)
    win.setWindowTitle('System Metrics Monitor')
    plot = win.addPlot(title="Temperatures")
    plot.setLabel('left', 'Temperature (°C)')
    plot.setLabel('bottom', 'Time (s)')
    plot.addLegend()
    plot.showGrid(x=True, y=True)

    # Create plot curves
    cpu_curve = plot.plot(pen='r', name='CPU Temp (°C)')
    gpu_curves = [plot.plot(pen='b', name=f'GPU Temp {i}-{x[0]} (°C)') for i, x in enumerate(current_metrics.gpu_info)]


    def update_plot():
        """Update the plot with filtered CPU/GPU temperatures, targeting 30 FPS."""
        global last_update
        current_time = time.time()
        elapsed = current_time - last_update

        # Update at ~30 FPS (0.0333 seconds)
        if elapsed < 1.0 / 30:
            return

        # Calculate visualization Hz
        viz_hz = 1.0 / elapsed if elapsed > 0 else 0.0
        last_update = current_time

        # Get filtered metrics
        current_metrics = metrics.get_filtered_metrics()
        times.append(current_time - start_time)

        # Update CPU temperature
        cpu_temp = max(t[2] for t in current_metrics.cpu_temp) if current_metrics.cpu_temp else 0.0
        cpu_temp_data.append(cpu_temp)

        # Update GPU temperature
        gpu_temp = [info[3] for info in current_metrics.gpu_info]
        for i,temp in enumerate(gpu_temp):
            gpu_temp_data[i].append(temp)
        # Update plot curves
        cpu_curve.setData(list(times), list(cpu_temp_data))
        for i,temp in enumerate(gpu_temp):
            gpu_curves[i].setData(list(times), list(gpu_temp_data[i]))

        # Log visualization Hz
        logging.info(f"Visualization Hz: {viz_hz:.2f}")


    # Set up QTimer for ~30 FPS updates
    timer = QTimer()
    timer.timeout.connect(update_plot)
    timer.start(int(1000/16))  # ~30 FPS (1000 ms / 30 ≈ 33.33 ms)

    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        metrics.stop()
        logging.info("Monitoring stopped")
    finally:
        metrics.stop()