# OsapiensTUMhackathon

## Quickstart (Colab or local)

Install dependencies:

```bash
python -m pip install -U pip
python -m pip install -r ONI-makeathon-challenge-2026-main/requirements.txt
```

Download the dataset from the public S3 bucket:

```bash
python -m src.download_data --local-dir ./data
```

This downloads `makeathon-challenge/` into `./data/`:

```text
data/makeathon-challenge/
```

## Notes

- The notebook walkthrough lives in [ONI-makeathon-challenge-2026-main/challenge.ipynb](ONI-makeathon-challenge-2026-main/challenge.ipynb).
- You can also use the Makefile in [ONI-makeathon-challenge-2026-main/Makefile](ONI-makeathon-challenge-2026-main/Makefile) for local setup.
