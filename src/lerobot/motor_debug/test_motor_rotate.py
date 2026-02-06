import math
import sys
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE

DEG_STEP = 10
TICKS_PER_DEG = 4095 / 360.0

ADDR_MIN_ANGLE_LIMIT = 0x09
ADDR_MAX_ANGLE_LIMIT = 0x0B

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
    
    print(f"\n[{port} / ID:{motor_id}] Connecting...")
    try:
        bus.connect(handshake=True)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    min_limit, _, _ = bus._read(ADDR_MIN_ANGLE_LIMIT, 2, motor_id)
    max_limit, _, _ = bus._read(ADDR_MAX_ANGLE_LIMIT, 2, motor_id)
    
    print(f"\n>>> Calibration Information <<<")
    print(f"Minimum angle limit (Min Limit): {min_limit}")
    print(f"Maximum angle limit (Max Limit): {max_limit}")
    print("--------------------------------")

    addr_pos, length_pos = STS_SMS_SERIES_CONTROL_TABLE["Present_Position"]
    addr_goal, length_goal = STS_SMS_SERIES_CONTROL_TABLE["Goal_Position"]

    current_pos, _, _ = bus._read(addr_pos, length_pos, motor_id)
    goal_pos = current_pos

    print(f"current pos: {current_pos}")
    print("remote: 'a' (-10˚), 'd' (+10˚), 'q' (exit)")

    while True:
        key = input(f"[goal:{goal_pos}] input(a/d/q): ").strip().lower()
        if key == "q":
            break
        
        next_goal = goal_pos
        if key == "a":
            next_goal -= int(round(DEG_STEP * TICKS_PER_DEG))
        elif key == "d":
            next_goal += int(round(DEG_STEP * TICKS_PER_DEG))
        else:
            continue

        if next_goal < min_limit:
            print(f"⚠ Warning: Reached the minimum limit({min_limit}).")
            next_goal = min_limit
        elif next_goal > max_limit:
            print(f"⚠ Warning: Reached the maximum limit({max_limit}).")
            next_goal = max_limit
            
        if next_goal != goal_pos:
            goal_pos = next_goal
            try:
                bus._write(addr_goal, length_goal, motor_id, goal_pos, raise_on_error=True)
                print(f"-> send order: {goal_pos}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("  (No movement: Reaching the limit)")

if __name__ == "__main__":
    main()
