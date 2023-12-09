import os
import subprocess
from datetime import datetime
from colorama import Fore, init

init(autoreset=True)

from friendsfamilytest.params import LOCAL_GIT_REPO
from friendsfamilytest.utils import time_it

repo_path = LOCAL_GIT_REPO


def get_current_branch():
    """Returns the name of the current Git branch."""
    branch = (
        subprocess.check_output(["git", "branch", "--show-current"]).strip().decode()
    )
    return branch


@time_it
def do_git_merge():
    os.chdir(repo_path)
    current_branch = get_current_branch()

    if current_branch == "master":
        perform_git_operations("master")
    else:
        perform_git_operations(current_branch)
        # Pulling the latest changes from master before merging
        subprocess.run(["git", "checkout", "master"])
        print(f"{Fore.BLUE}[+] Pulling latest changes from master")
        subprocess.run(["git", "pull", "origin", "master"])
        # Merging the current branch into master
        print(f"{Fore.BLUE}[+] Merging {current_branch} into master")
        subprocess.run(["git", "merge", current_branch])
        # Pushing the merged changes to master
        print(f"{Fore.BLUE}[+] Pushing merged changes to master")
        subprocess.run(["git", "push", "origin", "master"])
        # Switching back to the original branch
        print(f"{Fore.BLUE}[+] Switching back to {current_branch}")
        subprocess.run(["git", "checkout", current_branch])


def perform_git_operations(branch):
    print(f"{Fore.RED}[+] git status")
    subprocess.run(["git", "status", "."])

    print(f"{Fore.RED}[+] git add .")
    subprocess.run(["git", "add", "."])

    print(f"{Fore.RED}[+] git commit")
    current_timestamp = datetime.now()
    formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M")
    message = f"Automated commit via Python script - {formatted_timestamp}"
    subprocess.run(["git", "commit", "-m", message])

    print(f"{Fore.RED}[+] git push origin {branch}")
    subprocess.run(["git", "push", "origin", branch])


if __name__ == "__main__":
    do_git_merge()
