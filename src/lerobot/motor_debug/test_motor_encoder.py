from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE

def main():
    port_suffix = input("ttyACM port number (ex: 0, 1, 2): ").strip()
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
    
    print(f"[{port} / ID:{motor_id}] Connecting...")
    bus.connect(handshake=True)

    addr_pos, length_pos = STS_SMS_SERIES_CONTROL_TABLE["Present_Position"]

    print("Connected! Rotate the shaft manually. (Ctrl+C to exit)")

    try:
        while True:
            pos, _, _ = bus._read(addr_pos, length_pos, motor_id)
            print(f"Current Position: {pos}")
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()