import time
import subprocess

while True:
    subprocess.run(["python", "friendsfamilytest/data.py"])
    time.sleep(60)  # Wait for 7200 seconds = 2 hours

