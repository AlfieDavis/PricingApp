# generate_hash.py
# Interactive bcrypt hash generator for Streamlit-Authenticator credentials.
# Usage: run this file with Python. You'll be prompted for a username and
# password. The script will install passlib[bcrypt] automatically if
# necessary, generate a bcrypt hash for the password, and print out a
# ready-to-paste snippet for your Streamlit app credentials.

import sys
import subprocess
import json


def ensure_passlib_installed():
    """Ensure that the passlib[bcrypt] package is installed."""
    try:
        from passlib.hash import bcrypt  # noqa: F401
    except Exception:
        # Install passlib with bcrypt support via pip. We use subprocess here
        # because pip may not be available as an import in all environments.
        print("Installing passlib[bcrypt] ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "passlib[bcrypt]"])


def generate_hash():
    """Prompt the user for a username and password and output a credentials snippet."""
    try:
        from passlib.hash import bcrypt
    except Exception:
        print(
            "passlib is not installed. Installing now. If this fails, please install manually and rerun."
        )
        subprocess.check_call([sys.executable, "-m", "pip", "install", "passlib[bcrypt]"])
        from passlib.hash import bcrypt  # type: ignore

    print("\n=== Password → bcrypt hash generator ===\n")
    username = input("Username for snippet [admin]: ").strip() or "admin"
    display = input("Display name      [Admin User]: ").strip() or "Admin User"

    while True:
        try:
            import getpass
            pw1 = getpass.getpass("Enter password: ")
            pw2 = getpass.getpass("Confirm password: ")
        except Exception:
            # Fallback to regular input if getpass fails (e.g. in some IDEs)
            pw1 = input("Enter password: ")
            pw2 = input("Confirm password: ")
        if pw1 != pw2:
            print("Passwords do not match. Try again.\n")
            continue
        # Generate the bcrypt hash
        h = bcrypt.hash(pw1)
        print("\nHash:\n" + h)
        # Show a snippet the user can paste
        snippet = {username: {"name": display, "password": h}}
        print("\nPaste into app.py under credentials['usernames']:\n")
        print(json.dumps(snippet, indent=4))
        print("\n(Or just copy the hash string into the existing entry.)\n")
        again = input("Generate another? [y/N]: ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    ensure_passlib_installed()
    generate_hash()