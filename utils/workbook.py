"""Helpers for reading NTSA workbooks and their embedded mappings."""
from __future__ import annotations

import io

import pandas as pd


def _combine_excel_sheets(raw: bytes) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
    frames = []
    cause_code_mapping = None
    for sheet_name, frame in sheets.items():
        normalized_columns = {
            str(column).strip().lower().replace("_", " ")
            for column in frame.columns
        }
        if {"category", "code", "description"} <= normalized_columns:
            cause_code_mapping = frame.rename(
                columns={
                    column: str(column).strip().lower().replace(" ", "_")
                    for column in frame.columns
                }
            )
            continue
        frame = frame.copy()
        frame["source_sheet"] = str(sheet_name)
        frames.append(frame)
    if not frames:
        raise ValueError("The Excel workbook contains no worksheets.")
    return pd.concat(frames, ignore_index=True), cause_code_mapping
