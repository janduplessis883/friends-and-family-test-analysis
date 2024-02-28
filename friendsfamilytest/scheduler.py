import time
import subprocess
import sys

wait_time = 7200  # 2 hours

while True:
    subprocess.run(["python", "friendsfamilytest/data.py"])
    
    # Countdown for wait_time
    for remaining in range(wait_time, 0, -1):
        sys.stdout.write("\r")
        sys.stdout.write("{:2d} seconds remaining.".format(remaining))
        sys.stdout.flush()
        time.sleep(1)
    
    sys.stdout.write("\rExecuting script...                   \n")