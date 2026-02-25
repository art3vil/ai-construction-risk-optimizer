from pathlib import Path


def preprocess(input_path: str, output_path: str) -> None:
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if in_path.exists():
        out_path.write_bytes(in_path.read_bytes())
    else:
        out_path.write_text("")

