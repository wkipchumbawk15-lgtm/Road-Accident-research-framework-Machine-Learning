import pandas as pd
import pytest

from utils.preprocess_job import _combine_excel_sheets
from utils.research_schema import engineer_ntsa_features
from utils.train_model import load_dataset


def test_engineer_ntsa_features_normalizes_workbook_headers_and_maps_categories():
    raw = pd.DataFrame(
        [
            {
                "DATE": "2023-01-02",
                "DAY": "MONDAY",
                "TIME 24 HOURS": 830,
                "COUNTY": "NAIROBI",
                "ROAD": "RING ROAD",
                "PLACE": "CBD",
                "TYPE OF VEHICLE": "MOTOR CYCLE",
                "GENDER": "M",
                "AGE": 27,
                "CAUSE CODE": code,
                "VICTIM": "PED",
                "NO.": 1,
                "source_sheet": "2023",
            }
            for code in [1, 2, 7, 10]
        ]
    )

    engineered = engineer_ntsa_features(raw)

    assert {"date", "time_24_hours", "vehicle_type", "cause_code", "cause_category"} <= set(engineered.columns)
    assert engineered["month"].notna().all()
    assert engineered["day_of_week"].notna().all()
    assert engineered["hour"].notna().all()
    assert engineered["cause_category"].notna().all()
    assert len(engineered["cause_category"].unique()) == 3


def test_embedded_cause_code_mapping_is_used_and_mapping_sheet_is_not_accident_data(tmp_path):
    accident_rows = pd.DataFrame(
        [
            {
                "DATE": "2023-01-02",
                "DAY": "MONDAY",
                "TIME 24 HOURS": 830,
                "COUNTY": "NAIROBI",
                "ROAD": "RING ROAD",
                "PLACE": "CBD",
                "TYPE OF VEHICLE": "BUS",
                "GENDER": "M",
                "AGE": 27,
                "CAUSE CODE": 1,
                "VICTIM": "PED",
                "NO.": 1,
            },
            {
                "DATE": "2023-01-03",
                "DAY": "TUESDAY",
                "TIME 24 HOURS": 900,
                "COUNTY": "NAIROBI",
                "ROAD": "RING ROAD",
                "PLACE": "CBD",
                "TYPE OF VEHICLE": "BUS",
                "GENDER": "M",
                "AGE": 28,
                "CAUSE CODE": 2,
                "VICTIM": "PED",
                "NO.": 1,
            },
        ]
    )
    cause_codes = pd.DataFrame(
        [
            {"Category": "Driver Errors", "Code": 1, "Description": "Turning right"},
            {"Category": "Pedestrian Actions", "Code": 2, "Description": "Crossing road"},
        ]
    )
    workbook = tmp_path / "with_cause_codes.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        accident_rows.to_excel(writer, sheet_name="2023", index=False)
        cause_codes.to_excel(writer, sheet_name="NTSA Cause Codes", index=False)

    accident_data, mapping = _combine_excel_sheets(workbook.read_bytes())
    engineered = engineer_ntsa_features(
        accident_data,
        cause_code_mapping=mapping,
    )

    assert len(accident_data) == len(accident_rows)
    assert mapping is not None
    assert engineered["cause_description"].tolist() == ["Turning right", "Crossing road"]
    assert engineered["cause_category"].tolist() == ["Driver Errors", "Pedestrian Actions"]


def test_engineering_preserves_categories_already_assigned_by_uploaded_mapping(tmp_path):
    dataset = tmp_path / "mapped_categories.csv"
    pd.DataFrame(
        [
            {
                "date": "2023-01-02",
                "day": "MONDAY",
                "time_24_hours": "830",
                "county": "NAIROBI",
                "road": "RING ROAD",
                "place": "CBD",
                "vehicle_type": "BUS",
                "gender": "M",
                "age": 27,
                "victim": "PED",
                "no": 1,
                "cause_code": "1",
                "cause_description": "Turning right without due care",
                "cause_category": "Driver Errors",
            },
            {
                "date": "2023-01-03",
                "day": "TUESDAY",
                "time_24_hours": "900",
                "county": "NAIROBI",
                "road": "RING ROAD",
                "place": "CBD",
                "vehicle_type": "BUS",
                "gender": "M",
                "age": 28,
                "victim": "PED",
                "no": 1,
                "cause_code": "2",
                "cause_description": "Turning left without due care",
                "cause_category": "Pedestrian Actions",
            },
        ]
    ).to_csv(dataset, index=False)

    _, y, _, _ = load_dataset(str(dataset), "cause_category")

    assert set(y) == {"Driver Errors", "Pedestrian Actions"}


def test_load_dataset_engineers_required_features(tmp_path):
    dataset = tmp_path / "mini_ntsa.csv"
    pd.DataFrame(
        [
            {
                "date": "2023-01-02",
                "day": "MONDAY",
                "time_24_hours": "830",
                "county": "NAIROBI",
                "road": "RING ROAD",
                "place": "CBD",
                "vehicle_type": "MOTOR CYCLE",
                "gender": "M",
                "age": "27",
                "victim": "PED",
                "no": "12",
                "severity": "HIGH",
            },
            {
                "date": "2023-01-03",
                "day": "TUESDAY",
                "time_24_hours": "1735",
                "county": "NAKURU",
                "road": "OUTER RING",
                "place": "CITY",
                "vehicle_type": "BUS",
                "gender": "F",
                "age": "45",
                "victim": "DRIVER",
                "no": "17",
                "severity": "LOW",
            },
        ]
    ).to_csv(dataset, index=False)

    X, y, predictors, rows = load_dataset(str(dataset), "severity")

    assert rows == 2
    assert "hour" in X.columns
    assert "minute" in X.columns
    assert "day_of_week" in X.columns
    assert "severity" not in predictors
    assert set(y.unique()) == {"HIGH", "LOW"}


def test_load_dataset_can_use_provisional_cause_category(tmp_path):
    dataset = tmp_path / "cause_categories.csv"
    pd.DataFrame(
        [
            {
                "date": f"2023-01-0{index + 1}",
                "day": "MONDAY",
                "time_24_hours": "830",
                "county": "NAIROBI",
                "road": f"ROAD {index}",
                "place": f"PLACE {index}",
                "vehicle_type": "BUS",
                "gender": "M",
                "age": 27 + index,
                "victim": "PED",
                "no": 1,
                "cause_code": str(code),
            }
            for index, code in enumerate([1, 2, 7, 10])
        ]
    ).to_csv(dataset, index=False)

    X, y, predictors, rows = load_dataset(str(dataset), "cause_category")

    assert rows == 4
    assert set(y.unique()) == {
        "Human condition / impairment",
        "Speed / control",
        "Lane / manoeuvring",
    }
    assert "cause_code" not in predictors
    assert "cause_category" not in X.columns


def test_load_dataset_rejects_date_as_target_before_class_count_check(tmp_path):
    dataset = tmp_path / "many_dates.csv"
    pd.DataFrame(
        [
            {
                "date": f"2023-01-{index + 1:02d}",
                "day": "MONDAY",
                "time_24_hours": "830",
                "county": "NAIROBI",
                "road": "ROAD",
                "place": "PLACE",
                "vehicle_type": "BUS",
                "gender": "M",
                "age": 27,
                "victim": "PED",
                "no": 1,
                "severity": "HIGH" if index % 2 else "LOW",
            }
            for index in range(30)
        ]
    ).to_csv(dataset, index=False)

    with pytest.raises(ValueError, match="DATE.*cause_category"):
        load_dataset(str(dataset), "DATE")
