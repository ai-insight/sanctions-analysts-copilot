import os
import re
import subprocess

VERSION_FILE = "version.txt"


def read_version_file():
    if not os.path.exists(VERSION_FILE):
        raise FileNotFoundError(f"{VERSION_FILE} not found.")
    with open(VERSION_FILE, "r") as f:
        return f.read().strip()


def write_version_file(version):
    with open(VERSION_FILE, "w") as f:
        f.write(version + "\n")
    print(f"✅ {VERSION_FILE} updated to: {version}")


def get_clean_version():
    version = read_version_file()
    match = re.match(r"^(\d+\.\d+\.\d+)(?:_\d{14})?$", version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")
    return match.group(1)


def run_git_command(command, *args, capture_output=False):
    return subprocess.run(["git", command, *args], check=True, capture_output=capture_output, text=True)


def commit_version_file(clean_version):
    run_git_command("add", VERSION_FILE)
    run_git_command("commit", "-m", f"chore(release): finalize v{clean_version}")
    run_git_command("push")
    print("✅ Committed cleaned version.txt to main.")


def tag_exists(tag_name):
    result = run_git_command("tag", "--list", tag_name, capture_output=True)
    return tag_name in result.stdout


def create_git_tag(tag_name):
    if tag_exists(tag_name):
        print(f"⚠️ Tag {tag_name} already exists. Skipping tagging.")
        return
    run_git_command("tag", tag_name)
    run_git_command("push", "origin", tag_name)


def get_current_branch():
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()


def main():
    if get_current_branch() != "main":
        print("⚠️  Not on 'main' branch. Skipping finalization.")
        return

    clean_version = get_clean_version()
    write_version_file(clean_version)
    commit_version_file(clean_version)
    create_git_tag(f"v{clean_version}")
    print("🎉 Release finalized successfully!")


if __name__ == "__main__":
    main()
