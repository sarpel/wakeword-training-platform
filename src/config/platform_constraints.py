"""
Platform Constraints for Wakeword Detection
Defines RAM and Flash limits for common target platforms.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class PlatformConstraints:
    """Constraints for a specific hardware platform"""

    name: str
    max_flash_kb: int
    max_ram_kb: int
    description: str


# Predefined common platforms
PLATFORMS: Dict[str, PlatformConstraints] = {
    "esp32_s3": PlatformConstraints(
        name="ESP32-S3",
        max_flash_kb=4096,  # 4MB typical, but app partition might be smaller
        max_ram_kb=512,  # Internal SRAM
        description="Espressif ESP32-S3 with internal SRAM",
    ),
    "esp32_s3_psram": PlatformConstraints(
        name="ESP32-S3 (PSRAM)",
        max_flash_kb=8192,
        max_ram_kb=8192,  # 8MB external PSRAM
        description="Espressif ESP32-S3 with external PSRAM",
    ),
    "raspberry_pi_pico": PlatformConstraints(
        name="Raspberry Pi Pico",
        max_flash_kb=2048,
        max_ram_kb=264,
        description="RP2040 based microcontroller",
    ),
    "raspberry_pi_zero": PlatformConstraints(
        name="Raspberry Pi Zero",
        max_flash_kb=1024 * 1024,  # Virtually unlimited (SD card)
        max_ram_kb=512 * 1024,  # 512MB
        description="Raspberry Pi Zero / Zero W",
    ),
    "arduino_nano_33_ble": PlatformConstraints(
        name="Arduino Nano 33 BLE",
        max_flash_kb=1024,
        max_ram_kb=256,
        description="Nordic nRF52840 based board",
    ),
}


def get_platform_constraints(platform_name: str) -> PlatformConstraints:
    """
    Get constraints for a given platform.

    Args:
        platform_name: Key in PLATFORMS dictionary

    Returns:
        PlatformConstraints instance

    Raises:
        ValueError: If platform_name is not recognized
    """
    if platform_name not in PLATFORMS:
        raise ValueError(
            f"Unknown platform: {platform_name}. Available: {list(PLATFORMS.keys())}"
        )
    return PLATFORMS[platform_name]
