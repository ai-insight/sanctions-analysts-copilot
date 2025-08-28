import os
import sys
import subprocess
import re
from datetime import datetime

VERSION_FILE = "version.txt"
BUILD_METADATA_FILE = "build_metadata.txt"
IGNORE_REVS_FILE = ".git-blame-ignore-revs"
SEMVER_PATTERN = r"^(\d+)\.(\d+)\.(\d+)$"
README_PATH = "README.md"
CHANGELOG_PATH = "CHANGELOG.md"

def run_command(command, capture_output=True, check=True, text=True):
    result = subprocess.run(command, capture_output=capture_output, check=check, text=text)
    return result.stdout.strip() if capture_output else None

def read_file(file_path):
    with open(file_path, "r") as f:
        return f.read().strip()

def write_file(file_path, content):
    with open(file_path, "w") as f:
        f.write(content + "\n")

def append_to_file(file_path, content):
    with open(file_path, "a") as f:
        f.write(content)

def get_latest_commit_message():
    return run_command(["git", "log", "-1", "--pretty=%B"])

def get_latest_commit_hash():
    return run_command(["git", "rev-parse", "HEAD"])

def get_current_branch_name():
    return run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])

def bump_version(version, bump_type):
    match = re.match(SEMVER_PATTERN, version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")
    major, minor, patch = map(int, match.groups())
    if bump_type == "major":
        major, minor, patch = major + 1, 0, 0
    elif bump_type == "minor":
        minor, patch = minor + 1, 0
    elif bump_type == "fix":
        patch += 1
    else:
        raise ValueError(f"Unknown bump type: {bump_type}")
    return f"{major}.{minor}.{patch}"

def detect_bump_type(commit_msg, branch_name=None):
    msg = commit_msg.lower().strip()
    
    # Check branch name first if provided
    if branch_name:
        branch_lower = branch_name.lower().strip()
        # feat/ or feature/ branches can be major or minor
        if branch_lower.startswith(('feat/', 'feature/')):
            # Check commit message for breaking changes to determine major vs minor
            if "breaking change" in msg:
                return "major"
            else:
                return "minor"
        # fix/ branches are always patch/fix
        elif branch_lower.startswith('fix/'):
            return "fix"
    
    # Fall back to commit message analysis
    if "breaking change" in msg:
        return "major"
    elif re.search(r"^\s*feat", msg, re.IGNORECASE):
        return "minor"
    return "fix"

def update_ignore_revs(latest_hash):
    if not os.path.exists(IGNORE_REVS_FILE):
        open(IGNORE_REVS_FILE, "w").close()
    ignored_hashes = read_file(IGNORE_REVS_FILE).splitlines()
    if latest_hash not in ignored_hashes:
        append_to_file(IGNORE_REVS_FILE, latest_hash + "\n")
        run_command(["git", "add", IGNORE_REVS_FILE])

def update_readme(version_ts):
    timezone = datetime.now().astimezone().tzname()
    current_ts = datetime.now().strftime(f"%Y%m%d%H%M%S {timezone}")
    with open(README_PATH, "r") as f:
        lines = f.readlines()
    new_lines = [
        f"- **Version:** `{version_ts}`\n" if line.startswith("- **Version:** ") else
        f"- **Last Updated:** `{current_ts}`\n" if line.startswith("- **Last Updated:**") else
        line for line in lines
    ]
    write_file(README_PATH, "".join(new_lines))

def update_changelog(version_ts, semver, commit_msg):
    now = datetime.now().strftime("%Y-%m-%d")
    new_entry = f"\n| v{version_ts} | {commit_msg} | {now}|"
    if not os.path.exists(CHANGELOG_PATH):
        write_file(CHANGELOG_PATH, "# Changelog\n| version | Description | Date |\n| --- | --- | --- |")
    append_to_file(CHANGELOG_PATH, new_entry)

def commit_and_push(branch_name, new_version):
    # Add all files including README in a single commit
    run_command(["git", "add", VERSION_FILE, BUILD_METADATA_FILE, CHANGELOG_PATH, README_PATH])

    # Check if there are any changes to commit
    try:
        run_command(["git", "diff", "--staged", "--quiet"])
        print("🚫 No changes to commit. Version may already be up to date.")
        return
    except subprocess.CalledProcessError:
        # There are changes to commit, continue
        pass

    check_commit = f"chore(version): bump to v{new_version}"
    log_result = run_command(["git", "log", "--oneline", "-n", "5"])
    if check_commit in log_result:
        print("🚫 Version bump already exists in recent commits. Skipping commit.")
        return

    # Single commit with all changes
    run_command(["git", "commit", "-m", check_commit])
    latest_hash = get_latest_commit_hash()
    update_ignore_revs(latest_hash)
    
    # Add the ignore revs file if it was updated
    if os.path.exists(IGNORE_REVS_FILE):
        run_command(["git", "add", IGNORE_REVS_FILE])
        run_command(["git", "commit", "--amend", "--no-edit"])
    
    print(f"✅ Version bumped to v{new_version}")

def main():
    try:
        full_version = read_file(VERSION_FILE)
        current_version = full_version.split("_")[0]
        commit_msg = get_latest_commit_message()
        bump_type = detect_bump_type(commit_msg)
        new_version = bump_version(current_version, bump_type)

        current_ts = datetime.now().astimezone().strftime("%Y%m%d%H%M%S")
        new_version_ts = f"{new_version}_{current_ts}"
        write_file(VERSION_FILE, new_version_ts)
        write_file(BUILD_METADATA_FILE, new_version_ts)

        update_readme(new_version_ts)
        update_changelog(new_version_ts, new_version, commit_msg)
        commit_and_push(None, new_version)

    except Exception as e:
        print(f"❌ Version bump failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
