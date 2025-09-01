# generate_hash.py
# Interactive bcrypt hash generator for Streamlit-Authenticator credentials.
# Usage: run this file, type a username + password, copy the snippet it prints.

import sys, subprocess, json

# Ensure passlib[bcrypt] is available
try:
    from passlib.hash import bcrypt
except Exception:
    print("Installing passlib[bcrypt] ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "passlib[bcrypt]"])
    from passlib.hash import bcrypt

import getpass

def main():
    print("\n=== Password → bcrypt hash generator ===\n")
    username = input("Username for snippet [admin]: ").strip() or "admin"
    display  = input("Display name      [Admin User]: ").strip() or "Admin User"

    while True:
        pw1 = getpass.getpass("Enter password: ")
        pw2 = getpass.getpass("Confirm password: ")
        if pw1 != pw2:
            print("Passwords do not match. Try again.\n")
            continue

        h = bcrypt.hash(pw1)
        print("\nHash:\n" + h)

        # Minimal snippet you can paste into app.py under credentials['usernames']
        snippet = {
            username: {
                "name": display,
                "password": h
            }
        }
        print("\nPaste into app.py under credentials['usernames']:\n")
        print(json.dumps(snippet, indent=4))
        print("\n(Or just copy the hash string into the existing entry.)\n")

        again = input("Generate another? [y/N]: ").strip().lower()
        if again != "y":
            break

if __name__ == "__main__":
    main()
