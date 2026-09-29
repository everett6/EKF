"""Live rocket tracking and landing prediction.

An Extended Kalman Filter fuses GPS, barometric altitude and accelerometer
telemetry with a physics model (thrust, drag, parachutes, wind), and a
Monte Carlo predictor forecasts the rest of the flight and the landing
point from the current estimate, using weather that refreshes during the
flight.
"""

__version__ = "0.1.0"
