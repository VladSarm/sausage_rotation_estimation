import serial
import time
import argparse

class StepperController:
    def __init__(self, port, baud_rate=115200, timeout=1):
        """Initialize the stepper motor controller.
        
        Args:
            port (str): Serial port name (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baud_rate (int): Baud rate for serial communication
            timeout (float): Serial timeout in seconds
        """
        self.serial = serial.Serial(port, baud_rate, timeout=timeout)
        time.sleep(2)  # Wait for Arduino to reset
        
        # Clear any startup messages
        self.serial.flushInput()
        print(f"Connected to Arduino on {port}")
        
    def rotate_to_angle(self, angle, verbose=True):
        """Rotate the stepper motor to a specific angle.
        
        Args:
            angle (float): Target angle in degrees
            
        Returns:
            float: The final angle reported by Arduino, or None if failed
        """
        # if not (0 <= angle <= 360):
        #     print("Warning: Angle should be between 0 and 360 degrees")
        
        # Send command to Arduino
        command = f"ANGLE:{angle:.2f}\n"
        self.serial.write(command.encode())
        
        # Wait for feedback
        timeout_start = time.time()
        while time.time() < timeout_start + 10:  # 10-second timeout
            if self.serial.in_waiting > 0:
                response = self.serial.readline().decode().strip()
                if response.startswith("DONE:"):
                    final_angle = float(response[5:])
                    if verbose:
                        print(f"Rotation completed at {final_angle} degrees")
                    return final_angle
                elif response.startswith("ERROR:"):
                    print(f"Error: {response[6:]}")
                    return None
            time.sleep(0.1)
        
        print("Error: No response from Arduino")
        return None
    
    def close(self):
        """Close the serial connection."""
        if self.serial.is_open:
            self.serial.close()
            print("Serial connection closed")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Control stepper motor rotation')
    parser.add_argument('--port', type=str, default='/dev/tty.usbserial-1130',
                        help='Serial port for Arduino (default: /dev/tty.usbserial-1130)')
    
    args = parser.parse_args()
    
    controller = StepperController(args.port)  # Use '/dev/ttyUSB0' on Linux
    
    try:
        # Rotate to specific angles
        controller.rotate_to_angle(360-150)
        # time.sleep(1)
        # controller.rotate_to_angle(360*2)
        # time.sleep(1)
        # controller.rotate_to_angle(360*3)
        # time.sleep(1)
        # controller.rotate_to_angle(0)
        
    finally:
        # Always close the serial connection
        controller.close()