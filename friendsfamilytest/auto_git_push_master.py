import os
import time 
from datetime import datetime
import colorama 
import subprocess
from colorama import Fore, Back, Style, init
init(autoreset=True)

from friendsfamilytest.params import LOCAL_GIT_REPO

if __name__ == "__main__":
    start_time = time.time()
    print(f"\n{Fore.WHITE}{Back.BLACK}[AUTO] Git: Push to GitHub Repo")

    repo_path = LOCAL_GIT_REPO
    remote = "origin"
    branch = "master"
    os.chdir(repo_path)
    
    print(f"{Fore.RED}[+] git add")
    subprocess.run(["git", "add", "."])
    
    print(f"{Fore.RED}[+] git commit")
    current_timestamp = datetime.now()
    # Format the timestamp to include date, hour, and minute
    formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M")
    message = f"Automated commit via Python script - {formatted_timestamp}"
    subprocess.run(["git", "commit", "-m", message])
    
    print(f"{Fore.RED}[+] git push origin {branch}")
    subprocess.run(["git", "push", remote, branch])
    
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print(f"{Fore.BLACK}{Back.YELLOW}[✅] GIT PUSH SUCCESSFUL!")