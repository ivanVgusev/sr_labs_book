from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOXCELEB_ROOT = PROJECT_ROOT / "voxceleb" / "vox1"
VOX1_TEST_WAV = VOXCELEB_ROOT / "test" / "wav"
VOX1_DEV_WAV = VOXCELEB_ROOT / "dev" / "wav"
VOX1_META = VOXCELEB_ROOT / "meta"


def require_voxceleb(*paths: Path) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "VoxCeleb is not prepared. Missing paths:\n"
            f"{formatted}\n"
            "Extract the archives under sr_labs_book/voxceleb/vox1 first."
        )
