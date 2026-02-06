import sys
import time
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE

DEG_STEP = 20 
ACCELERATION = 50   
TICKS_PER_DEG = 4095 / 360.0

ADDR_ACCELERATION = 0x29
ADDR_LOAD = 0x3C
ADDR_CURRENT = 0x45
ADDR_STATUS = 0x41

def main():
    port_suffix = input("ttyACM port number (ex: 0): ").strip()
    motor_id = int(input("motor ID (ex: 1): ").strip())
    port = f"/dev/ttyACM{port_suffix}"

    motors = {"m": Motor(motor_id, "sts3215", MotorNormMode.RANGE_M100_100)}
    bus = FeetechMotorsBus(port=port, motors=motors)
    
    print(f"\n[{port} / ID:{motor_id}] Connecting...")
    bus.connect(handshake=False)

    bus._write(ADDR_ACCELERATION, 1, motor_id, ACCELERATION, raise_on_error=True)

    addr_pos, length_pos = STS_SMS_SERIES_CONTROL_TABLE["Present_Position"]
    addr_goal, length_goal = STS_SMS_SERIES_CONTROL_TABLE["Goal_Position"]

    current_pos, _, _ = bus._read(addr_pos, length_pos, motor_id)
    goal_pos = current_pos

    print(f"\ncurrent pos: {current_pos}")
    print("remote: 'a'(-20˚), 'd'(+20˚), 'q'(exit)")

    while True:
        try:
            key = input(f"\n[Goal:{goal_pos} / Curr:{current_pos}] input > ").strip().lower()
            if key == "q":
                break
            
            if key == "a":
                goal_pos -= int(round(DEG_STEP * TICKS_PER_DEG))
            elif key == "d":
                goal_pos += int(round(DEG_STEP * TICKS_PER_DEG))
            else:
                pass

            goal_pos = max(0, min(4095, goal_pos))

            bus._write(addr_goal, length_goal, motor_id, goal_pos, raise_on_error=True)

            for _ in range(5):
                curr, _, _ = bus._read(addr_pos, length_pos, motor_id)
                load, _, _ = bus._read(ADDR_LOAD, 2, motor_id)
                status, _, _ = bus._read(ADDR_STATUS, 1, motor_id)
                
                print(f"  -> moving.. Pos:{curr}, Load:{load}, Status:{status}")
                
                if status != 0:
                    print(f"  Error (Status: {status})")
                
                time.sleep(0.1)

            current_pos = curr

        except Exception as e:
            print(f"Comunnication error: {e}")
            break

if __name__ == "__main__":
    main()
