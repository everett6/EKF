import time
import csv
import math
import serial
import numpy as np
import pynmea2
import pymap3d
from numpy.linalg import inv
from mpu9250_jmdev.mpu_9250 import MPU9250
from mpu9250_jmdev.registers import *

# === CONFIGURATION ===
PRIMARY_GPS = "/dev/ttyACM0"
BACKUP_GPS = "/dev/ttyACM1"
GPS_BAUD = 38400
ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT = 37.395390, -121.851837, 180
LOG_FILE = "sensor_log.csv"

# === INITIALIZE IMU ===
print("Initializing IMU...")
imu = MPU9250(
    address_ak=AK8963_ADDRESS,
    address_mpu_master=MPU9050_ADDRESS_68,
    address_mpu_slave=None,
    bus=1,
    gfs=GFS_1000,
    afs=AFS_8G,
    mfs=AK8963_BIT_16,
    mode=AK8963_MODE_C100HZ
)
imu.configure()

# === GPS ===
def init_gps(port):
    try:
        return serial.Serial(port, GPS_BAUD, timeout=0.5)
    except:
        return None

gps = init_gps(PRIMARY_GPS)
gps_fallback = init_gps(BACKUP_GPS)
gps_fail_count = 0

def get_gps():
    global gps, gps_fail_count
    if not gps: return None
    try:
        while gps.in_waiting:
            line = gps.readline().decode("ascii", errors="ignore").strip()
            if "$GGA" in line:
                msg = pynmea2.parse(line)
                if int(msg.gps_qual) > 0:
                    gps_fail_count = 0
                    lat = msg.latitude if msg.lat_dir == 'N' else -msg.latitude
                    lon = msg.longitude if msg.lon_dir == 'E' else -msg.longitude
                    alt = float(msg.altitude)
                    return lat, lon, alt
    except:
        gps_fail_count += 1
        if gps_fail_count > 5 and gps_fallback:
            gps = gps_fallback
            gps_fail_count = 0
    return None

# === EKF ===
def init_ekf():
    x = np.zeros(12)
    P = np.eye(12) * 100
    Q = np.eye(12) * 0.1
    R = np.eye(12)
    R[0:3] *= 5.0  # GPS
    R[3:6] *= 0.1  # Accel
    R[6:9] *= 0.2  # Vel
    R[9:12] *= 0.1  # Orientation
    return x, P, Q, R

def ekf_predict(x, P, Q, dt):
    F = np.eye(12)
    F[0:3, 3:6] = np.eye(3) * dt
    F[3:6, 6:9] = np.eye(3) * dt
    return F @ x, F @ P @ F.T + Q

def ekf_update(xp, Pp, z, R):
    H = np.eye(12)
    y = z - H @ xp
    y[11] = math.atan2(math.sin(y[11]), math.cos(y[11]))  # normalize yaw
    S = H @ Pp @ H.T + R
    K = Pp @ H.T @ inv(S)
    x = xp + K @ y
    P = (np.eye(12) - K @ H) @ Pp
    return x, P

def get_orientation(ax, ay, az, mx, my, mz):
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, np.sqrt(ay**2 + az**2))
    yaw = math.atan2(
        my * np.cos(roll) - mz * np.sin(roll),
        mx * np.cos(pitch) + my * np.sin(roll) * np.sin(pitch) + mz * np.cos(roll) * np.sin(pitch)
    )
    return np.array([roll, pitch, yaw])

# === MAIN ===
x, P, Q, R = init_ekf()
vel_est = np.zeros(3)
accel_filtered = np.zeros(3)
last_time = time.time()

with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "time", "lat", "lon", "alt",
        "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz",
        "roll", "pitch", "yaw",
        "ekf_x", "ekf_y", "ekf_z",
        "ekf_vx", "ekf_vy", "ekf_vz"
    ])

    print("Logging data... CTRL+C to stop.")
    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now

            gps_data = get_gps()
            if not gps_data: continue
            lat, lon, alt = gps_data
            enu = pymap3d.geodetic2enu(lat, lon, alt, ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)

            acc = np.array(imu.readAccelerometerMaster())
            gyro = np.array(imu.readGyroscopeMaster())
            mag = np.array(imu.readMagnetometerMaster())
            accel_filtered = 0.8 * accel_filtered + 0.2 * acc
            vel_est += accel_filtered * dt
            orientation = get_orientation(*accel_filtered, *mag)

            z = np.concatenate([enu, accel_filtered, vel_est, orientation])
            xp, Pp = ekf_predict(x, P, Q, dt)
            x, P = ekf_update(xp, Pp, z, R)

            writer.writerow([
                round(now, 3), lat, lon, alt,
                *acc, *gyro, *mag,
                *orientation,
                x[0], x[1], x[2],
                x[3], x[4], x[5]
            ])
            f.flush()

    except KeyboardInterrupt:
        print("Logging stopped.")
