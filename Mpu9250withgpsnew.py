import math, serial, datetime
import time
import string
import math
import serial.tools.list_ports
from mpu9250_jmdev.registers import* 
from mpu9250_jmdev.mpu_9250 import MPU9250
import pynmea2
import sys
import pymap3d
import numpy as np
from scipy.linalg import block_diag
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

#Define the devices
gps_port = "/dev/ttyACM0"  # Change this to your GPS port
ser = serial.Serial(gps_port, baudrate=38400, timeout=0.5)
mpu = MPU9250(
    address_ak= AK8963_ADDRESS,
    address_mpu_master=MPU9050_ADDRESS_68,
    address_mpu_slave= None,
    bus=1,
    gfs= GFS_1000,
    afs=AFS_8G,
    mfs=AK8963_BIT_16,
    mode=AK8963_MODE_C100HZ)

def init_ekf(dt):
    """Initialize EKF state and matrices"""
    x = np.zeros(9)  # State vector [x, y, z, vx, vy, vz, ax, ay, az]
    P = np.eye(9) * 100  # Initial state covariance
    
    # Process noise
    Q = np.eye(9)
    Q[0:3, 0:3] *= 0.1  # Position noise
    Q[3:6, 3:6] *= 0.2  # Velocity noise
    Q[6:9, 6:9] *= 0.3  # Acceleration noise
    
    # Measurement noise
    R = np.eye(6)
    R[0:3, 0:3] *= 5.0  # GPS measurement noise
    R[3:6, 3:6] *= 0.1  # IMU measurement noise
    
    return x, P, Q, R

def predict_step(x, P, Q, dt):
    """Predict step of the EKF"""
    # State transition matrix
    F = np.eye(9)
    F[0:3, 3:6] = np.eye(3) * dt
    F[3:6, 6:9] = np.eye(3) * dt
    
    # Predict state and covariance
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    
    return x_pred, P_pred

def update_step(x_pred, P_pred, measurement, R):
    """Update step of the EKF"""
    # Measurement matrix
    H = np.zeros((6, 9))
    H[0:3, 0:3] = np.eye(3)  # Position measurements
    H[3:6, 6:9] = np.eye(3)  # Acceleration measurements
    
    # Innovation
    y = measurement - H @ x_pred
    
    # Innovation covariance
    S = H @ P_pred @ H.T + R
    
    # Kalman gain
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    # Update state and covariance
    x = x_pred + K @ y
    P = (np.eye(9) - K @ H) @ P_pred
    
    return x, P


# Initialize variables
epsilon = sys.float_info.epsilon
roll = 0
pitch = 0
alpha = 0.80 # must be from 0 to 1 
filtered_acceleration_x = 0
filtered_acceleration_y = 0
filtered_acceleration_z = 0
yaw = 0

# Initialize the serial connection for the GPS
dataout = pynmea2.NMEAStreamReader()
newdata=ser.readline()



def calculate_accel_data():
    try:
        acceleration_data = mpu.readAccelerometerMaster()
        if acceleration_data is None:
            return 0, 0, 0
            
        acceleration_x = float(acceleration_data['x']) if 'x' in acceleration_data else 0
        acceleration_y = float(acceleration_data['y']) if 'y' in acceleration_data else 0
        acceleration_z = float(acceleration_data['z']) if 'z' in acceleration_data else 0
        
        return acceleration_x, acceleration_y, acceleration_z
    except:
        return 0, 0, 0

def get_gps_data():
    try:
        newdata = ser.readline().decode()
        if newdata[0:6] == "$GPRMC":
            newmsg = pynmea2.parse(newdata)
            lat = float(newmsg.latitude) if newmsg.latitude else 0
            lng = float(newmsg.longitude) if newmsg.longitude else 0
            return lat, lng
        return 0, 0
    except:
        return 0, 0

def calculate_rotation(x, y, z):
    try:
        pitch = math.atan2(x, math.sqrt(y*y + z*z)) * 180 / math.pi
        roll = math.atan2(y, math.sqrt(x*x + z*z)) * 180 / math.pi
        yaw = math.atan2(z, math.sqrt(x*x + y*y)) * 180 / math.pi
        return yaw, pitch, roll
    except:
        return 0, 0, 0
def low_pass_filter(previous_value, new_value, alpha=0.8):
    """Apply a low-pass filter to smooth the acceleration data."""
    return alpha * previous_value + (1 - alpha) * new_value


# Initialize EKF
dt = 0.1  # Time step (10 Hz)
x, P, Q, R = init_ekf(dt)
last_time = time.time()

# Main loop
while True:
    try:
        # Get GPS data
        lat, lng = get_gps_data()
        
        # Get accelerometer data
        accel_x, accel_y, accel_z = calculate_accel_data()
        
        # Apply low pass filter
        filtered_acceleration_x = low_pass_filter(filtered_acceleration_x, accel_x)
        filtered_acceleration_y = low_pass_filter(filtered_acceleration_y, accel_y)
        filtered_acceleration_z = low_pass_filter(filtered_acceleration_z, accel_z)
        
        # Simple output
        print(f"GPS: {lat}, {lng}")
        print(f"Accel: {accel_x}, {accel_y}, {accel_z}")
        print(f"Filtered: {filtered_acceleration_x}, {filtered_acceleration_y}, {filtered_acceleration_z}")
        print("-" * 20)
        
        time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(0.1)
        continue


