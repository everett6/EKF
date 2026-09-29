import time
import csv
import serial
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import platform
import pynmea2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
from mpu9250_jmdev.mpu_9250 import MPU9250
from mpu9250_jmdev.registers import *
import datetime
import os

# --- Config ---
GPS_PORT = "/dev/ttyACM0"
GPS_BAUD = 115200

ACCEL_BIAS = [-0.01083754, -0.0222767, -1.45536608]
GYRO_BIAS = [0.37492605, -0.35068089, 0.17542196]
MAG_OFFSET = [39.14789512, 10.96897845, 7.34605759]
MAG_SCALE = [1.01685423, 0.93676252, 1.05366477]

UBX_SET_RATE_10HZ = bytes([
    0xB5, 0x62,
    0x06, 0x08,
    0x06, 0x00,
    0x64, 0x00,  # 100 decimal = 10Hz
    0x01, 0x00,
    0x01, 0x00,
    0x7A, 0x12
])

# --- Initialize IMU ---
imu = MPU9250(
    address_ak=AK8963_ADDRESS,
    address_mpu_master=MPU9050_ADDRESS_68,
    bus=1,
    gfs=GFS_1000,
    afs=AFS_8G,
    mfs=AK8963_BIT_16,
    mode=AK8963_MODE_C100HZ
)
imu.configure()

# --- Initialize GPS ---
try:
    gps = serial.Serial(GPS_PORT, GPS_BAUD, timeout=1)
    gps.write(UBX_SET_RATE_10HZ)
except Exception as e:
    gps = None
    print(f"GPS Init failed: {e}")

# --- Calibration ---
def calibrate_accel(accel_raw):
    return [accel_raw[0] - ACCEL_BIAS[0],
            accel_raw[1] - ACCEL_BIAS[1],
            accel_raw[2] - ACCEL_BIAS[2]]

def calibrate_gyro(gyro_raw):
    return [gyro_raw[0] - GYRO_BIAS[0],
            gyro_raw[1] - GYRO_BIAS[1],
            gyro_raw[2] - GYRO_BIAS[2]]

def calibrate_mag(mag_raw):
    return [(mag_raw[0] - MAG_OFFSET[0]) * MAG_SCALE[0],
            (mag_raw[1] - MAG_OFFSET[1]) * MAG_SCALE[1],
            (mag_raw[2] - MAG_OFFSET[2]) * MAG_SCALE[2]]

# --- GPS data and lock ---
gps_data = {
    "lat": None, "lon": None, "alt": None, "fix": False,
    "satellites": 0, "gps_time": None, "gps_time_delta": None
}
gps_lock = threading.Lock()

def parse_gps():
    global gps_data
    if gps is None or not gps.in_waiting:
        return
    try:
        while gps.in_waiting:
            line = gps.readline().decode("ascii", errors="ignore").strip()
            if line.startswith('$GPRMC') or line.startswith('$GNRMC'):
                try:
                    msg = pynmea2.parse(line)
                    if msg.status == 'A':
                        # pynmea2's latitude/longitude are already signed (S and W negative).
                        lat = msg.latitude
                        lon = msg.longitude
                        # Convert GPS date/time to timestamp
                        gps_epoch = None
                        try:
                            if msg.datestamp is not None and msg.timestamp is not None:
                                dt = datetime.datetime.combine(msg.datestamp, msg.timestamp)
                                gps_epoch = dt.timestamp()
                        except Exception:
                            gps_epoch = None
                        sys_time = time.time()
                        delta = sys_time - gps_epoch if gps_epoch is not None else None
                        with gps_lock:
                            gps_data.update({
                                "lat": lat, "lon": lon, "fix": True,
                                "gps_time": gps_epoch, "gps_time_delta": delta
                            })
                except Exception as e:
                    print(f"RMC parse error: {e}")
            elif line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                try:
                    msg = pynmea2.parse(line)
                    if int(msg.gps_qual) > 0:
                        # pynmea2's latitude/longitude are already signed (S and W negative).
                        lat = msg.latitude
                        lon = msg.longitude
                        alt = float(msg.altitude)
                        sats = int(msg.num_sats)
                        # Use last known gps_epoch from RMC if available
                        gps_epoch = gps_data.get("gps_time")
                        delta = time.time() - gps_epoch if gps_epoch is not None else None
                        with gps_lock:
                            gps_data.update({
                                "lat": lat, "lon": lon, "alt": alt, "fix": True,
                                "satellites": sats, "gps_time": gps_epoch,
                                "gps_time_delta": delta
                            })
                except Exception as e:
                    print(f"GGA parse error: {e}")
    except Exception as e:
        print(f"General GPS parse error: {e}")

# --- Global flags ---
logging_active = False
log_thread = None

# --- Log filename ---
LOG_FILE = "/home/everett/Programs/Data/datalogger.csv"

# --- Update rate tracking ---
imu_prev_time = None
gps_prev_time = None
imu_hz = 0.0
gps_hz = 0.0

# --- Data buffers for plotting ---
MAX_LEN = 150
mag_xs, mag_ys, mag_zs = deque(maxlen=MAX_LEN), deque(maxlen=MAX_LEN), deque(maxlen=MAX_LEN)
accel_xs, accel_ys, accel_zs = deque(maxlen=MAX_LEN), deque(maxlen=MAX_LEN), deque(maxlen=MAX_LEN)
gyro_xs, gyro_ys, gyro_zs = deque(maxlen=MAX_LEN), deque(maxlen=MAX_LEN), deque(maxlen=MAX_LEN)
times = deque(maxlen=MAX_LEN)

def logging_loop():
    global logging_active, LOG_FILE, imu_prev_time, gps_prev_time, imu_hz, gps_hz
    with open(LOG_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "sys_time",
            "mag_sys_time", "mag_x", "mag_y", "mag_z",
            "accel_sys_time", "accel_x", "accel_y", "accel_z",
            "gyro_sys_time", "gyro_x", "gyro_y", "gyro_z",
            "gps_sys_time", "gps_internal_time", "gps_lat", "gps_lon", "gps_alt", "gps_fix", "gps_sats"
        ])

        while logging_active:
            try:
                sys_time = time.time()

                # Read magnetometer with timestamp and calibrate
                mag_sys_time = time.time()
                mag_raw = imu.readMagnetometerMaster()
                mag = calibrate_mag(mag_raw)

                # IMU update rate calculation
                global imu_prev_time, imu_hz
                if imu_prev_time is not None:
                    dt_imu = mag_sys_time - imu_prev_time
                    if dt_imu > 0:
                        imu_hz = 1.0 / dt_imu
                imu_prev_time = mag_sys_time

                # Read accelerometer with timestamp and calibrate
                accel_sys_time = time.time()
                accel_raw = imu.readAccelerometerMaster()
                accel = calibrate_accel(accel_raw)

                # Read gyroscope with timestamp and calibrate
                gyro_sys_time = time.time()
                gyro_raw = imu.readGyroscopeMaster()
                gyro = calibrate_gyro(gyro_raw)

                # Parse GPS data and calculate GPS update rate
                parse_gps()
                with gps_lock:
                    lat = gps_data["lat"]
                    lon = gps_data["lon"]
                    alt = gps_data["alt"] if gps_data["alt"] else 0.0
                    fix = gps_data["fix"]
                    sats = gps_data.get("satellites", 0)
                    gps_internal_time = gps_data.get("gps_time")  # GPS epoch time from NMEA
                    gps_sys_time = time.time()

                global gps_prev_time, gps_hz
                if gps_prev_time is not None and gps_sys_time is not None:
                    dt_gps = gps_sys_time - gps_prev_time
                    if dt_gps > 0:
                        gps_hz = 1.0 / dt_gps
                gps_prev_time = gps_sys_time

                # Write CSV row
                writer.writerow([
                    sys_time,
                    mag_sys_time, mag[0], mag[1], mag[2],
                    accel_sys_time, accel[0], accel[1], accel[2],
                    gyro_sys_time, gyro[0], gyro[1], gyro[2],
                    gps_sys_time, gps_internal_time, lat, lon, alt, fix, sats
                ])
                file.flush()

                # Append to plotting buffers
                times.append(sys_time)
                mag_xs.append(mag[0])
                mag_ys.append(mag[1])
                mag_zs.append(mag[2])
                accel_xs.append(accel[0])
                accel_ys.append(accel[1])
                accel_zs.append(accel[2])
                gyro_xs.append(gyro[0])
                gyro_ys.append(gyro[1])
                gyro_zs.append(gyro[2])

                time.sleep(0.1)  # target 10Hz
            except Exception as e:
                print(f"Logging error: {e}")
                time.sleep(0.5)

# --- GUI Setup ---
root = tk.Tk()
root.title("Sensor Logger UI")
root.geometry("1000x600")

left_frame = ttk.Frame(root, width=300)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

right_frame = ttk.Frame(root)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

start_btn = ttk.Button(left_frame, text="Start Logging")
stop_btn = ttk.Button(left_frame, text="Stop Logging")
exit_btn = ttk.Button(left_frame, text="Exit")
view_csv_btn = ttk.Button(left_frame, text="View CSV")
clear_csv_btn = ttk.Button(left_frame, text="Clear CSV")

filename_label = ttk.Label(left_frame, text="Log Filename:")
filename_entry = ttk.Entry(left_frame)
filename_entry.insert(0, LOG_FILE)

start_btn.pack(fill=tk.X, pady=2)
stop_btn.pack(fill=tk.X, pady=2)
view_csv_btn.pack(fill=tk.X, pady=2)
clear_csv_btn.pack(fill=tk.X, pady=2)
exit_btn.pack(fill=tk.X, pady=20)
filename_label.pack(pady=(20, 0))
filename_entry.pack(fill=tk.X)

gps_label = ttk.Label(left_frame, text="GPS Data", font=("Helvetica", 14, "bold"))
gps_label.pack(pady=(20, 5))

gps_lat_label = ttk.Label(left_frame, text="Latitude: --")
gps_lat_label.pack(anchor=tk.W)
gps_lon_label = ttk.Label(left_frame, text="Longitude: --")
gps_lon_label.pack(anchor=tk.W)
gps_alt_label = ttk.Label(left_frame, text="Altitude: --")
gps_alt_label.pack(anchor=tk.W)
gps_fix_label = ttk.Label(left_frame, text="Fix Status: No")
gps_fix_label.pack(anchor=tk.W)
gps_sats_label = ttk.Label(left_frame, text="Satellites: 0")
gps_sats_label.pack(anchor=tk.W)

gps_hz_label = ttk.Label(left_frame, text="GPS Update Rate: 0.0 Hz")
gps_hz_label.pack(anchor=tk.W)
imu_hz_label = ttk.Label(left_frame, text="IMU Update Rate: 0.0 Hz")
imu_hz_label.pack(anchor=tk.W)

gps_ts_label = ttk.Label(left_frame, text="GPS Timestamp: --")
gps_ts_label.pack(anchor=tk.W)
sys_ts_label = ttk.Label(left_frame, text="System Timestamp: --")
sys_ts_label.pack(anchor=tk.W)
gps_delta_label = ttk.Label(left_frame, text="GPS-System Delta: --")
gps_delta_label.pack(anchor=tk.W)

fig, axs = plt.subplots(3, 1, figsize=(7, 7), constrained_layout=True)

axs[0].set_title("Magnetometer (μT)")
axs[0].set_ylabel("μT")
mag_lines = [axs[0].plot([], [], label=axis)[0] for axis in ["X", "Y", "Z"]]
axs[0].legend(loc='upper right')
axs[0].grid(True)

axs[1].set_title("Accelerometer (m/s²)")
axs[1].set_ylabel("m/s²")
accel_lines = [axs[1].plot([], [], label=axis)[0] for axis in ["X", "Y", "Z"]]
axs[1].legend(loc='upper right')
axs[1].grid(True)

axs[2].set_title("Gyroscope (deg/s)")
axs[2].set_ylabel("deg/s")
axs[2].set_xlabel("Samples")
gyro_lines = [axs[2].plot([], [], label=axis)[0] for axis in ["X", "Y", "Z"]]
axs[2].legend(loc='upper right')
axs[2].grid(True)

canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Variables to avoid jittery label updates
last_gps_ts_display = None
last_sys_ts_display = None
last_delta_display = None
last_gps_hz_display = None
last_imu_hz_display = None

def update_gps_labels():
    global last_gps_ts_display, last_sys_ts_display, last_delta_display, last_gps_hz_display, last_imu_hz_display
    with gps_lock:
        fix = gps_data["fix"]
        lat = gps_data["lat"]
        lon = gps_data["lon"]
        alt = gps_data["alt"]
        sats = gps_data.get("satellites", 0)
        gps_ts_val = gps_data.get("gps_time")
        delta_val = gps_data.get("gps_time_delta")

    sys_ts_val = time.time()
    gps_hz_val = gps_hz
    imu_hz_val = imu_hz

    if fix:
        gps_lat_label.config(text=f"Latitude: {lat:.6f}")
        gps_lon_label.config(text=f"Longitude: {lon:.6f}")
        gps_alt_label.config(text=f"Altitude: {alt:.2f} m" if alt else "Altitude: 0.00 m")
        gps_fix_label.config(text="Fix Status: Yes")
        gps_sats_label.config(text=f"Satellites: {sats}")
    else:
        gps_lat_label.config(text="Latitude: --")
        gps_lon_label.config(text="Longitude: --")
        gps_alt_label.config(text="Altitude: --")
        gps_fix_label.config(text="Fix Status: No")
        gps_sats_label.config(text="Satellites: 0")

    if last_gps_hz_display is None or abs(gps_hz_val - last_gps_hz_display) > 0.1:
        gps_hz_label.config(text=f"GPS Update Rate: {gps_hz_val:.1f} Hz")
        last_gps_hz_display = gps_hz_val

    if last_imu_hz_display is None or abs(imu_hz_val - last_imu_hz_display) > 0.1:
        imu_hz_label.config(text=f"IMU Update Rate: {imu_hz_val:.1f} Hz")
        last_imu_hz_display = imu_hz_val

    if gps_ts_val is not None:
        if last_gps_ts_display is None or abs(gps_ts_val - last_gps_ts_display) > 0.001:
            gps_ts_label.config(text=f"GPS Timestamp: {gps_ts_val:.3f}")
            last_gps_ts_display = gps_ts_val
    else:
        gps_ts_label.config(text="GPS Timestamp: --")

    if last_sys_ts_display is None or abs(sys_ts_val - last_sys_ts_display) > 0.001:
        sys_ts_label.config(text=f"System Timestamp: {sys_ts_val:.3f}")
        last_sys_ts_display = sys_ts_val

    if delta_val is not None:
        if last_delta_display is None or abs(delta_val - last_delta_display) > 0.001:
            gps_delta_label.config(text=f"GPS-System Delta: {delta_val:.3f} s")
            last_delta_display = delta_val
    else:
        gps_delta_label.config(text="GPS-System Delta: --")

    root.after(500, update_gps_labels)

def update_plots():
    if not root.winfo_exists():
        return

    if len(times) < 2:
        root.after(200, update_plots)
        return

    xdata = list(range(len(times)))

    def set_line_data(lines, data_lists, ax):
        for i, line in enumerate(lines):
            ydata = list(data_lists[i])
            # Make sure lengths match xdata, truncate if needed:
            if len(ydata) > len(xdata):
                ydata = ydata[-len(xdata):]
            elif len(ydata) < len(xdata):
                # pad ydata with last value or zeros if needed
                ydata = ([0] * (len(xdata) - len(ydata))) + ydata
            line.set_data(xdata, ydata)
        ax.relim()
        ax.autoscale_view()

    set_line_data(mag_lines, [mag_xs, mag_ys, mag_zs], axs[0])
    set_line_data(accel_lines, [accel_xs, accel_ys, accel_zs], axs[1])
    set_line_data(gyro_lines, [gyro_xs, gyro_ys, gyro_zs], axs[2])

    canvas.draw()
    root.after(200, update_plots)


def start_logging():
    global logging_active, log_thread, LOG_FILE
    if logging_active:
        messagebox.showwarning("Already Logging", "Logging is already running.")
        return
    filename = filename_entry.get().strip()
    if not filename:
        messagebox.showerror("Invalid Filename", "Please enter a valid filename.")
        return
    LOG_FILE = filename
    logging_active = True
    log_thread = threading.Thread(target=logging_loop, daemon=True)
    log_thread.start()
    start_btn.config(state=tk.DISABLED)
    stop_btn.config(state=tk.NORMAL)
    print(f"Started logging to {LOG_FILE}")

def stop_logging():
    global logging_active, log_thread
    if not logging_active:
        messagebox.showwarning("Not Logging", "Logging is not currently running.")
        return
    logging_active = False
    if log_thread is not None:
        log_thread.join(timeout=2)
    start_btn.config(state=tk.NORMAL)
    stop_btn.config(state=tk.DISABLED)
    print("Stopped logging")

def view_csv():
    filename = filename_entry.get().strip()
    if not filename:
        messagebox.showerror("Invalid Filename", "Please enter a valid filename.")
        return
    if not os.path.exists(filename):
        messagebox.showerror("File Not Found", f"File {filename} does not exist.")
        return
    system = platform.system()
    try:
        if system == "Linux":
            subprocess.Popen(["xdg-open", filename])
        elif system == "Darwin":
            subprocess.Popen(["open", filename])
        elif system == "Windows":
            os.startfile(filename)
        else:
            messagebox.showerror("Unsupported OS", f"Cannot open file on {system}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open file: {e}")

def clear_csv():
    filename = filename_entry.get().strip()
    if not filename:
        messagebox.showerror("Invalid Filename", "Please enter a valid filename.")
        return
    if not os.path.exists(filename):
        messagebox.showwarning("File Not Found", f"File {filename} does not exist.")
        return
    confirm = messagebox.askyesno("Confirm Clear", f"Are you sure you want to clear all data in '{filename}'?")
    if not confirm:
        return
    try:
        with open(filename, "w") as file:
            pass  # Truncate file
        messagebox.showinfo("Cleared", f"File {filename} cleared.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to clear file: {e}")

def on_exit():
    global logging_active
    if messagebox.askokcancel("Quit", "Do you really want to quit?"):
        logging_active = False
        root.quit()
        root.destroy()

# Bind buttons
start_btn.config(command=start_logging)
stop_btn.config(command=stop_logging)
view_csv_btn.config(command=view_csv)
clear_csv_btn.config(command=clear_csv)
exit_btn.config(command=on_exit)

stop_btn.config(state=tk.DISABLED)

root.protocol("WM_DELETE_WINDOW", on_exit)

update_gps_labels()
update_plots()

root.mainloop()
