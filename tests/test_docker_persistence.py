"""
Test Docker Volume Persistence
This script verifies that files created inside the Docker container
persist on the host machine via mounted volumes.
"""

import os
import subprocess
import time
from pathlib import Path


def test_persistence():
    # 1. Create a test file inside the container
    test_file_name = f"docker_test_{int(time.time())}.txt"
    host_path = Path("data") / test_file_name

    print(f"[*] Creating test file via docker-compose exec...")
    try:
        # Use 'dashboard' service to write to /app/data
        subprocess.run(
            ["docker-compose", "exec", "-T", "dashboard", "touch", f"/app/data/{test_file_name}"], check=True
        )

        # 2. Check if it exists on the host
        time.sleep(1)  # Give it a second to sync
        if host_path.exists():
            print(f"[OK] Persistence verified! File found at {host_path}")
            # Cleanup
            host_path.unlink()
        else:
            print(f"[FAIL] File not found on host at {host_path}")

    except subprocess.CalledProcessError:
        print("[!] Error: Could not execute command in container. Is docker-compose up running?")
    except FileNotFoundError:
        print("[!] Error: docker-compose command not found.")


if __name__ == "__main__":
    test_persistence()
