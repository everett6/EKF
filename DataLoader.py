import csv
import numpy as np
from datetime import datetime
from math import cos, sin, sqrt, radians

def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    # WGS84 constants
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    # Convert degrees to radians
    lat, lon, lat0, lon0 = map(radians, [lat, lon, lat0, lon0])
    # Reference point ECEF
    x0 = (a * cos(lat0) * cos(lon0))
    y0 = (a * cos(lat0) * sin(lon0))
    z0 = (b * sin(lat0))
    # Point ECEF
    x = (a * cos(lat) * cos(lon))
    y = (a * cos(lat) * sin(lon))
    z = (b * sin(lat))
    # Delta
    dx, dy, dz = x - x0, y - y0, alt - alt0
    # ENU conversion
    east = -sin(lon0) * dx + cos(lon0) * dy
    north = -sin(lat0) * cos(lon0) * dx - sin(lat0) * sin(lon0) * dy + cos(lat0) * dz
    up = cos(lat0) * cos(lon0) * dx + cos(lat0) * sin(lon0) * dy + sin(lat0) * dz
    return east, north, up

global_data_store = []
global_loader = None


class SensorData:
    def __init__(self, timestamp, **kwargs):
        self.timestamp = timestamp
        for key, value in kwargs.items():
            setattr(self, key, value)
        # ENU conversion if gps_lat/gps_lon/gps_alt present
        lat = getattr(self, 'gps_lat', None)
        lon = getattr(self, 'gps_lon', None)
        alt = getattr(self, 'gps_alt', None)
        if lat is not None and lon is not None and alt is not None and lat !='' and lon!='' and alt!= '0':
            # Reference location (customize as needed)
            lat0, lon0, alt0 = float(lat), float(lon), float(alt)  # Use first entry as reference
            try:
                east, north, up = geodetic_to_enu(float(lat), float(lon), float(alt), lat0, lon0, alt0)
                self.enu = (east, north, up)
            except Exception:
                self.enu = None
        else:
            self.enu = None

    def __repr__(self):
        attrs = ', '.join(f"{k}={v}" for k, v in self.__dict__.items() if k != 'timestamp')
        return f"[{self.timestamp}] {attrs}"

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.global_data_store = []
        self.current_index = -1


    def load_data(self):
        with open(self.file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                ts = row.get('Time', None)
                if ts:
                    try:
                        ts = float(ts)
                    except ValueError:
                        ts = str(ts)
                # Accept latitude, longitude, altitude as field names
                data = {k: float(v) if self._is_float(v) else v for k, v in row.items() if k != 'Time'}
                sensor_entry = SensorData(timestamp=ts, **data)
                self.global_data_store.append(sensor_entry)
            
    def render_data(self):
        self.current_index += 1
        if self.current_index > len(self.global_data_store):
            raise ValueError("No more data to render")
        return self.global_data_store[self.current_index]

    def data_size(self):
        return len(self.global_data_store)
        
    def _is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False


     

