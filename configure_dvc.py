import os
import subprocess
import sys

from dotenv import load_dotenv


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Load environment variables from .env
    load_dotenv(os.path.join(project_root, ".env"))

    user = os.getenv("HESSENBOX_USER")
    password = os.getenv("HESSENBOX_PASSWORD")
    dir = os.getenv("HESSENBOX_DIR", "EcoTwin/dvc_poc_tests/")

    if not user or not password:
        print(
            "Error: HESSENBOX_USER and HESSENBOX_PASSWORD must be defined in your local .env file!"
        )
        sys.exit(1)

    url = f"webdavs://next.hessenbox.de/remote.php/dav/files/{user}/{dir}"

    print(f"Configuring DVC remote locally for user '{user}'...")

    try:
        # Add the remote locally (-d sets default, -f forces overwrite if existing)
        subprocess.run(
            [
                "uv",
                "run",
                "dvc",
                "remote",
                "add",
                "--local",
                "-d",
                "-f",
                "hessenbox",
                url,
            ],
            check=True,
            cwd=project_root,
        )
        subprocess.run(
            [
                "uv",
                "run",
                "dvc",
                "remote",
                "modify",
                "--local",
                "hessenbox",
                "user",
                user,
            ],
            check=True,
            cwd=project_root,
        )
        subprocess.run(
            [
                "uv",
                "run",
                "dvc",
                "remote",
                "modify",
                "--local",
                "hessenbox",
                "password",
                password,
            ],
            check=True,
            cwd=project_root,
        )
        print("\nDVC configuration complete!")
        print("You can now download the models by running: uv run dvc pull")
    except subprocess.CalledProcessError as e:
        print(f"\nDVC configuration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
