"""Foreground launcher for the packaged Streamlit app on all IPv4 interfaces."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch AutoExplain locally (Ctrl+C stops it).")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    try:
        from streamlit.web import cli
    except ImportError:
        parser.exit(1, "From the AutoExplain repository checkout, install the studio extra: python -m pip install -e '.[studio]'\n")
    import sys
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("studio_app.py")),
                "--server.address=0.0.0.0", f"--server.port={args.port}",
                "--server.headless=true", "--server.maxUploadSize=10",
                "--browser.gatherUsageStats=false"]
    return cli.main()


if __name__ == "__main__":
    main()
