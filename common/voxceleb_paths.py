from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "data"

VOX1_TEST_WAV = PROJECT_ROOT / "voxceleb1_test" / "wav"
VOX1_DEV_WAV = PROJECT_ROOT / "voxceleb1_dev" / "wav"
VOX1_META = PROJECT_ROOT / "voxceleb_meta"

VOX1_TEST_WAV_MP3 = VOX1_TEST_WAV.parent / "wav_mp3_codec"
VOX1_DEV_WAV_MP3 = VOX1_DEV_WAV.parent / "wav_mp3_codec"

VOX1_VERI_TEST2 = VOX1_META / "veri_test2.txt"
VOX1_IDEN_SPLIT = VOX1_META / "iden_split.txt"


def require_voxceleb(*paths: Path) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "VoxCeleb is not prepared. Missing paths:\n"
            f"{formatted}\n"
            "Extract the archives under sr_labs_book/voxceleb/vox1 first."
        )
