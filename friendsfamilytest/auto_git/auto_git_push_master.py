import os
import time 
from datetime import datetime
import colorama 
import subprocess
from colorama import Fore, Back, Style, init
init(autoreset=True)

from friendsfamilytest.params import LOCAL_GIT_REPO
from friendsfamilytest.utils import *

repo_path = LOCAL_GIT_REPO

@time_it
def do_git_push():
    
    remote = "origin"
    branch = "master"
    os.chdir(repo_path)
    
    print(f"{Fore.RED}[+] git status")
    subprocess.run(["git", "status", "."])
    
    print(f"{Fore.RED}[+] git add .")
    subprocess.run(["git", "add", "."])
    
    print(f"{Fore.RED}[+] git commit")
    current_timestamp = datetime.now()
    # Format the timestamp to include date, hour, and minute
    formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M")
    message = f"Automated commit via Python script - {formatted_timestamp}"
    subprocess.run(["git", "commit", "-m", message])
    
    print(f"{Fore.RED}[+] git push origin {branch}")
    subprocess.run(["git", "push", remote, branch])
    

if __name__ == "__main__":
    do_git_push()