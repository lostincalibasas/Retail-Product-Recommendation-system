"""Convert Complete Journey CSVs to Parquet and write a data-quality report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


EXPECTED_TABLES = (
    "campaign_desc",
    "campaign_table",
    "causal_data",
    "coupon",
    "coupon_redempt",
    "hh_demographic",
    "product",
    "transaction_data",
)

RELATIONSHIPS = (
    ("transaction_data", "household_key", "hh_demographic", "household_key"),
    ("transaction_data", "PRODUCT_ID", "product", "PRODUCT_ID"),
    ("coupon_redempt", "household_key", "hh_demographic", "household_key"),
    ("coupon_redempt", "COUPON_UPC", "coupon", "COUPON_UPC"),
    ("coupon", "PRODUCT_ID", "product", "PRODUCT_ID"),
    ("coupon", "CAMPAIGN", "campaign_desc", "CAMPAIGN"),
    ("campaign_table", "CAMPAIGN", "campaign_desc", "CAMPAIGN"),
    ("causal_data", "PRODUCT_ID", "product", "PRODUCT_ID"),
)

STREAM_DTYPES = {
    "causal_data": {
        "PRODUCT_ID": "Int64",
        "STORE_ID": "Int64",
        "WEEK_NO": "Int64",
        "display": "string",
        "mailer": "string",
    },
    "transaction_data": {
        "household_key": "Int64",
        "BASKET_ID": "Int64",
        "DAY": "Int64",
        "PRODUCT_ID": "Int64",
        "QUANTITY": "float64",
        "SALES_VALUE": "float64",
        "STORE_ID": "Int64",
        "RETAIL_DISC": "float64",
        "TRANS_TIME": "Int64",
        "WEEK_NO": "Int64",
        "COUPON_DISC": "float64",
        "COUPON_MATCH_DISC": "float64",
    },
}


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def stream_table(path: Path, output_path: Path, table_name: str) -> tuple[dict, dict | None]:
    row_count = 0
    missing_by_column = {}
    writer = None
    transaction_values = None
    if table_name == "transaction_data":
        transaction_values = {
            "households": set(),
            "stores": set(),
            "products": set(),
            "days": set(),
            "day_min": None,
            "day_max": None,
            "missing_sales_value": 0,
        }

    try:
        chunks = pd.read_csv(
            path,
            chunksize=250_000,
            dtype=STREAM_DTYPES[table_name],
        )
        for frame in chunks:
            if not missing_by_column:
                missing_by_column = {column: 0 for column in frame.columns}
            missing = frame.isna().sum()
            for column, count in missing.items():
                missing_by_column[column] += int(count)

            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            row_count += len(frame)

            if transaction_values is not None:
                transaction_values["households"].update(frame["household_key"].dropna().unique())
                transaction_values["stores"].update(frame["STORE_ID"].dropna().unique())
                transaction_values["products"].update(frame["PRODUCT_ID"].dropna().unique())
                transaction_values["days"].update(frame["DAY"].dropna().unique())
                valid_days = frame["DAY"].dropna()
                if not valid_days.empty:
                    low, high = int(valid_days.min()), int(valid_days.max())
                    transaction_values["day_min"] = low if transaction_values["day_min"] is None else min(transaction_values["day_min"], low)
                    transaction_values["day_max"] = high if transaction_values["day_max"] is None else max(transaction_values["day_max"], high)
                transaction_values["missing_sales_value"] += int(frame["SALES_VALUE"].isna().sum())
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError(f"Source table is empty: {path}")
    summary = {
        "rows": row_count,
        "columns": len(missing_by_column),
        "missing_cells": sum(missing_by_column.values()),
        "missing_by_column": {
            column: count for column, count in missing_by_column.items() if count
        },
    }
    return summary, transaction_values


def inspect_and_convert(raw_dir: Path, processed_dir: Path) -> dict:
    missing_files = [name for name in EXPECTED_TABLES if not (raw_dir / f"{name}.csv").is_file()]
    if missing_files:
        names = ", ".join(f"{name}.csv" for name in missing_files)
        raise FileNotFoundError(f"Missing source table(s) in {raw_dir}: {names}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    metrics = None
    for name in EXPECTED_TABLES:
        source_path = raw_dir / f"{name}.csv"
        output_path = processed_dir / f"{name}.parquet"
        if name in STREAM_DTYPES:
            tables[name], transaction_values = stream_table(source_path, output_path, name)
            if transaction_values is not None:
                metrics = {
                    "transaction_rows": tables[name]["rows"],
                    "households_in_transactions": len(transaction_values["households"]),
                    "stores": len(transaction_values["stores"]),
                    "skus_in_transactions": len(transaction_values["products"]),
                    "transaction_day_min": transaction_values["day_min"],
                    "transaction_day_max": transaction_values["day_max"],
                    "transaction_day_count": len(transaction_values["days"]),
                    "transaction_missing_sales_value": transaction_values["missing_sales_value"],
                }
        else:
            frame = read_table(source_path)
            frame.to_parquet(output_path, index=False)
            tables[name] = {
                "rows": len(frame),
                "columns": len(frame.columns),
                "missing_cells": int(frame.isna().sum().sum()),
                "missing_by_column": {
                    column: int(count)
                    for column, count in frame.isna().sum().items()
                    if count
                },
            }

    product_ids = set(pd.read_parquet(processed_dir / "product.parquet", columns=["PRODUCT_ID"])["PRODUCT_ID"].dropna())
    household_ids = set(pd.read_parquet(processed_dir / "hh_demographic.parquet", columns=["household_key"])["household_key"].dropna())
    coupon_ids = set(pd.read_parquet(processed_dir / "coupon.parquet", columns=["COUPON_UPC"])["COUPON_UPC"].dropna())
    campaigns = set(pd.read_parquet(processed_dir / "campaign_desc.parquet", columns=["CAMPAIGN"])["CAMPAIGN"].dropna())
    metrics["households_with_demographics"] = len(household_ids)
    parent_keys = {
        ("hh_demographic", "household_key"): household_ids,
        ("product", "PRODUCT_ID"): product_ids,
        ("coupon", "COUPON_UPC"): coupon_ids,
        ("campaign_desc", "CAMPAIGN"): campaigns,
    }

    relationships = []
    for child_table, child_column, parent_table, parent_column in RELATIONSHIPS:
        child = pd.read_parquet(
            processed_dir / f"{child_table}.parquet", columns=[child_column]
        )[child_column]
        keys = parent_keys[(parent_table, parent_column)]
        valid = child.notna()
        unmatched = valid & ~child.isin(keys)
        relationships.append(
            {
                "child": f"{child_table}.{child_column}",
                "parent": f"{parent_table}.{parent_column}",
                "rows_checked": len(child),
                "null_key_rows": int((~valid).sum()),
                "unmatched_rows": int(unmatched.sum()),
                "unmatched_distinct_keys": int(child[unmatched].nunique()),
            }
        )

    return {"tables": tables, "metrics": metrics, "relationships": relationships}


def render_report(results: dict, report_path: Path) -> None:
    tables = results["tables"]
    metrics = results["metrics"]
    relationships = results["relationships"]
    transaction_demographic_link = next(
        relation for relation in relationships
        if relation["child"] == "transaction_data.household_key"
    )

    lines = [
        "# Complete Journey: Data Exploration (Phase 1)",
        "",
        "Generated from the local source CSVs by `scripts/ingest_data.py`. Counts below are computed from this dataset copy, not copied from a model or estimate.",
        "",
        "## Dataset overview",
        "",
        f"- Transaction rows: {metrics['transaction_rows']:,}",
        f"- Households represented in transactions: {metrics['households_in_transactions']:,}",
        f"- Households with a demographics record: {metrics['households_with_demographics']:,}",
        f"- Distinct stores: {metrics['stores']:,}",
        f"- Distinct purchased product IDs (SKU proxy): {metrics['skus_in_transactions']:,}",
        f"- Transaction `DAY` index: {metrics['transaction_day_min']:,} to {metrics['transaction_day_max']:,} ({metrics['transaction_day_count']:,} distinct values)",
        f"- Transaction rows with missing `SALES_VALUE`: {metrics['transaction_missing_sales_value']:,}",
        "",
        "`DAY` is the dataset's relative integer day index, not a calendar date. The CSV schema provides no date origin, so this report does not translate it into calendar dates. Product IDs are used as the SKU count proxy because the source provides no separate SKU key.",
        "",
        "## Source table sizes and missingness",
        "",
        "| Source table | Rows | Columns | Missing cells | Columns with missing values |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name in EXPECTED_TABLES:
        table = tables[name]
        columns = ", ".join(
            f"`{column}` ({count:,})"
            for column, count in table["missing_by_column"].items()
        ) or "None"
        lines.append(
            f"| `{name}` | {table['rows']:,} | {table['columns']} | {table['missing_cells']:,} | {columns} |"
        )

    lines.extend(
        [
            "",
            "## Relationship checks",
            "",
            "Checks compare non-null child keys to distinct keys in the referenced source table. An unmatched row is a key not found in that parent table; this is a diagnostic, not an assumption that every table must be a strict relational dimension.",
            "",
            "| Child key | Parent key | Rows checked | Null child keys | Unmatched rows | Distinct unmatched keys |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for relation in relationships:
        lines.append(
            f"| `{relation['child']}` | `{relation['parent']}` | {relation['rows_checked']:,} | {relation['null_key_rows']:,} | {relation['unmatched_rows']:,} | {relation['unmatched_distinct_keys']:,} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation and limits",
            "",
            f"- Demographics cover {metrics['households_with_demographics']:,} households, versus {metrics['households_in_transactions']:,} in transactions. The {transaction_demographic_link['unmatched_distinct_keys']:,} unmatched transaction household keys therefore reflect partial demographic coverage, not missing transaction records.",
            "- `transaction_data` records household-product transactions and observed `SALES_VALUE`; it does not contain sales-representative recommendations or acceptance labels.",
            "- `STORE_ID` is counted directly from transactions; the provided eight-file extract does not include a store dimension for validating store IDs against a separate table.",
            "- Missing values and key mismatches are reported as observed. No rows are silently dropped or imputed by this Phase 1 ingestion.",
            "- Dataset page: [Kaggle Complete Journey](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey). License listed there: [DbCL-1.0](http://opendatacommons.org/licenses/dbcl/1.0/). Review the license terms for the database contents before redistribution or other use.",
            "- Download used for this run: [GitHub release v1.0-data](https://github.com/bltap-plmarket/dunnhumby-complete-journey/releases/tag/v1.0-data). The archive SHA-256 was checked against the release's published digest before extraction.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    results = inspect_and_convert(root / "data" / "raw", root / "data" / "processed")
    report = root / "reports" / "data_exploration.md"
    render_report(results, report)

    print("Converted CSV tables to Parquet:")
    for name, summary in results["tables"].items():
        print(f"  {name}: {summary['rows']:,} rows, {summary['missing_cells']:,} missing cells")
    print(f"\nExploration report: {report}")
    print(f"Transactions: {results['metrics']['transaction_rows']:,} rows; "
          f"{results['metrics']['stores']:,} stores; "
          f"{results['metrics']['skus_in_transactions']:,} product IDs; "
          f"DAY {results['metrics']['transaction_day_min']}–{results['metrics']['transaction_day_max']}")


if __name__ == "__main__":
    main()
