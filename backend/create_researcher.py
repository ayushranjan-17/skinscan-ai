"""Create a researcher account. Run on the server, from the repo root:

    python3 -m backend.create_researcher "Full Name" researcher@example.org

The password is asked for without echoing. Researchers can read the biopsy retraining queue and export it,
so this role cannot be chosen at public sign-up. Uses the same database as the server (SKINSCAN_DB_PATH).
"""
import getpass
import os
import sys

# This command never issues login tokens, so the server's signing secret is not needed here.
os.environ.setdefault("SKINSCAN_ALLOW_DEV_SECRET", "1")

from backend.database import DB_PATH, create_user, init_db  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip())
        return 2
    name, email = sys.argv[1], sys.argv[2]
    password = getpass.getpass("Password (at least 10 characters): ")
    if password != getpass.getpass("Repeat password: "):
        print("Passwords do not match.")
        return 1
    init_db()
    try:
        user = create_user(name=name, email=email, password=password, role="researcher")
    except ValueError as e:
        print(f"Could not create the account: {e}")
        return 1
    print(f"Researcher account created for {user['email']} in {DB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
