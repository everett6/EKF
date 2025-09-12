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



def get_gps_data():
    global ser, newdata
    """
    Reads GPS data from the serial port and parses it.

    Returns:
        A tuple (latitude, longitude) in degrees, or (None, None) if parsing fails.
    """
    # Read a line of data from the serial port and decode it
    newdata = ser.readline().decode()

    # Check if the data starts with the $GPRMC identifier
    if newdata[0:6] == "$GPRMC":
        # Parse the NMEA sentence
        newmsg = pynmea2.parse(newdata)
        lat = newmsg.latitude
        lng = newmsg.longitude

        # Return the latitude and longitude
        return lat, lng
    else:
        # If the data is not $GPRMC, return None for both latitude and longitude
        return None, None



def calculate_accel_data():
    acceleration_data = mpu.readAccelerometerMaster()
    acceleration_x = acceleration_data['x']
    acceleration_y = acceleration_data['y']
    acceleration_z = acceleration_data['z']
    
    return acceleration_x, acceleration_y, acceleration_z

def calculate_rotation(x, y, z):
    pitch = math.atan2(x, math.sqrt(y**2 + z**2)) * 180 / math.pi
    roll = math.atan2(y, math.sqrt(x**2 + z**2)) * 180 / math.pi
    yaw = math.atan2(z, math.sqrt(x**2 + y**2)) * 180 / math.pi
    return yaw, pitch, roll

def calculate_pitch(acceleration_y, acceleration_z):
    pitchnotdegree = math.atan(acceleration_y / acceleration_z+epsilon)
    pitch = pitchnotdegree / (2 * math.pi) * 360
    return pitch

def calculate_roll(acceleration_x, acceleration_z):
    rollnotdeg = math.atan(acceleration_x / acceleration_z)
    roll = rollnotdeg / (2 * math.pi) * 360
    return roll

def low_pass_filter(prev_value, new_value):
    filtered_value = alpha * prev_value + (1 - alpha) * new_value
    return filtered_value

def calculate_angular_velocity():
    gyro_data = mpu.readGyroscopeMaster()
    angular_velocity_x = gyro_data['x']
    angular_velocity_y = gyro_data['y']
    angular_velocity_z = gyro_data['z']
    
    return angular_velocity_x, angular_velocity_y, angular_velocity_z





def get_gps_backup():
    newdata = ser.readline().decode()
    if newdata[0:6] =="$GPRMC":
        newmsg=pynmea2.parse(newdata)
                lat=newmsg.latitude
        lng=newmsg.longitude
        return lat, lng
    
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
        acceleration_x, acceleration_y, acceleration_z = calculate_accel_data()
        
        # Apply low pass filter
        filtered_acceleration_x = low_pass_filter(filtered_acceleration_x, acceleration_x)
        filtered_acceleration_y = low_pass_filter(filtered_acceleration_y, acceleration_y)
        filtered_acceleration_z = low_pass_filter(filtered_acceleration_z, acceleration_z)
        
        # Update time
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        if lat is not None and lng is not None:
            # Convert GPS to local ENU coordinates
            x_enu, y_enu, z_enu = pymap3d.geodetic2enu(lat, lng, 180, 37.395390, -121.851837, 180)
            gps_pos = np.array([x_enu, y_enu, z_enu])
            
            # Combine GPS and IMU measurements
            measurement = np.concatenate((
                gps_pos,
                np.array([filtered_acceleration_x, filtered_acceleration_y, filtered_acceleration_z])
            ))
            
            # EKF predict and update
            x_pred, P_pred = predict_step(x, P, Q, dt)
            x, P = update_step(x_pred, P_pred, measurement, R)
            
            # Print basic output
            print(f"GPS: {lat}, {lng}")
            print(f"ENU: {x_enu}, {y_enu}, {z_enu}")
            print(f"Accel: {acceleration_x, acceleration_y, acceleration_z}")
            print(f"Position: {x[0]}, {x[1]}, {x[2]}")
            print(f"Velocity: {x[3]}, {x[4]}, {x[5]}")
            print(f"Accel_EKF: {x[6]}, {x[7]}, {x[8]}")
            print("-" * 20)
            
        time.sleep(0.1)  # 10Hz update rate
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue


