# Complete Journey: Data Exploration (Phase 1)

Generated from the local source CSVs by `scripts/ingest_data.py`. Counts below are computed from this dataset copy, not copied from a model or estimate.

## Dataset overview

- Transaction rows: 2,595,732
- Households represented in transactions: 2,500
- Households with a demographics record: 801
- Distinct stores: 582
- Distinct purchased product IDs (SKU proxy): 92,339
- Transaction `DAY` index: 1 to 711 (711 distinct values)
- Transaction rows with missing `SALES_VALUE`: 0

`DAY` is the dataset's relative integer day index, not a calendar date. The CSV schema provides no date origin, so this report does not translate it into calendar dates. Product IDs are used as the SKU count proxy because the source provides no separate SKU key.

## Source table sizes and missingness

| Source table | Rows | Columns | Missing cells | Columns with missing values |
| --- | ---: | ---: | ---: | --- |
| `campaign_desc` | 30 | 4 | 0 | None |
| `campaign_table` | 7,208 | 3 | 0 | None |
| `causal_data` | 36,786,524 | 5 | 0 | None |
| `coupon` | 124,548 | 3 | 0 | None |
| `coupon_redempt` | 2,318 | 4 | 0 | None |
| `hh_demographic` | 801 | 8 | 0 | None |
| `product` | 92,353 | 7 | 0 | None |
| `transaction_data` | 2,595,732 | 12 | 0 | None |

## Relationship checks

Checks compare non-null child keys to distinct keys in the referenced source table. An unmatched row is a key not found in that parent table; this is a diagnostic, not an assumption that every table must be a strict relational dimension.

| Child key | Parent key | Rows checked | Null child keys | Unmatched rows | Distinct unmatched keys |
| --- | --- | ---: | ---: | ---: | ---: |
| `transaction_data.household_key` | `hh_demographic.household_key` | 2,595,732 | 0 | 1,168,429 | 1,699 |
| `transaction_data.PRODUCT_ID` | `product.PRODUCT_ID` | 2,595,732 | 0 | 0 | 0 |
| `coupon_redempt.household_key` | `hh_demographic.household_key` | 2,318 | 0 | 462 | 123 |
| `coupon_redempt.COUPON_UPC` | `coupon.COUPON_UPC` | 2,318 | 0 | 0 | 0 |
| `coupon.PRODUCT_ID` | `product.PRODUCT_ID` | 124,548 | 0 | 0 | 0 |
| `coupon.CAMPAIGN` | `campaign_desc.CAMPAIGN` | 124,548 | 0 | 0 | 0 |
| `campaign_table.CAMPAIGN` | `campaign_desc.CAMPAIGN` | 7,208 | 0 | 0 | 0 |
| `causal_data.PRODUCT_ID` | `product.PRODUCT_ID` | 36,786,524 | 0 | 0 | 0 |

## Interpretation and limits

- Demographics cover 801 households, versus 2,500 in transactions. The 1,699 unmatched transaction household keys therefore reflect partial demographic coverage, not missing transaction records.
- `transaction_data` records household-product transactions and observed `SALES_VALUE`; it does not contain sales-representative recommendations or acceptance labels.
- `STORE_ID` is counted directly from transactions; the provided eight-file extract does not include a store dimension for validating store IDs against a separate table.
- Missing values and key mismatches are reported as observed. No rows are silently dropped or imputed by this Phase 1 ingestion.
- Dataset page: [Kaggle Complete Journey](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey). License listed there: [DbCL-1.0](http://opendatacommons.org/licenses/dbcl/1.0/). Review the license terms for the database contents before redistribution or other use.
- Download used for this run: [GitHub release v1.0-data](https://github.com/bltap-plmarket/dunnhumby-complete-journey/releases/tag/v1.0-data). The archive SHA-256 was checked against the release's published digest before extraction.
