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


def perform_git_operations(branch):
    print(f"{Fore.GREEN}[+] git status")
    subprocess.run(["git", "status", "."])

    print(f"{Fore.GREEN}[+] git add .")
    subprocess.run(["git", "add", "."])

    print(f"{Fore.GREEN}[+] git commit")
    current_timestamp = datetime.now()
    formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M")
    message = f"Automated commit via Python script - {formatted_timestamp}"
    subprocess.run(["git", "commit", "-m", message])

    print(f"{Fore.GREEN}[+] git push origin {branch}")
    subprocess.run(["git", "push", "origin", branch])


@time_it
def push_changes_to_github():
    os.chdir(repo_path)
    current_branch = get_current_branch()

    # Pulling the latest changes from the current branch
    print(f"{Fore.GREEN}[+] Pulling latest changes from origin/{current_branch}")
    subprocess.run(["git", "pull", "origin", current_branch])

    perform_git_operations(current_branch)


if __name__ == "__main__":
    push_changes_to_github()
