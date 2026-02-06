import time
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE

def main():
    port_suffix = input("ttyACM port number (ex: 0): ").strip()
    if not port_suffix.isdigit():
        print("input number.")
        return
    port = f"/dev/ttyACM{port_suffix}"

    motor_id_str = input("motor ID (ex: 1): ").strip()
    if not motor_id_str.isdigit():
        print("input number.")
        return
    motor_id = int(motor_id_str)

    motors = {"m": Motor(motor_id, "sts3215", MotorNormMode.RANGE_M100_100)}
    bus = FeetechMotorsBus(port=port, motors=motors)
    bus.connect(handshake=False)

    ADDR_VOLTAGE = 0x3E
    ADDR_LOAD = 0x3C

    print(f"Start monitoring on {port}, Motor ID: {motor_id}")

    try:
        while True:
            volts_raw, _, _ = bus._read(ADDR_VOLTAGE, 1, motor_id)
            load, _, _ = bus._read(ADDR_LOAD, 2, motor_id)
            
            voltage = volts_raw / 10.0
            print(f"Voltage: {voltage}V | Load: {load}")
            
            if voltage < 10.0:
                print("Warning: Low Voltage!")
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()