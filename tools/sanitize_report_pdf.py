"""Copy a PDF while removing document metadata and hidden metadata streams."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(source)
    writer = PdfWriter(clone_from=reader)
    writer.metadata = None
    writer.root_object.pop("/Metadata", None)

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        writer.write(stream)
    os.replace(temporary, destination)


if __name__ == "__main__":
    main()
