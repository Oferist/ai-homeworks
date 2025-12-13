from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df, df=df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats


### HW03 CHANGE: New test for heuristics ###
def test_new_heuristics():
    """Проверка новых эвристик: константы, нули, дубли ID."""
    df = pd.DataFrame({
        'id': [1, 2, 1],         # Дубликат ID (1)
        'const': [5, 5, 5],      # Константная колонка
        'zeros': [0, 0, 10],     # 66% нулей
        'normal': [1, 2, 3]
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Передаем df, так как новые эвристики зависят от него
    flags = compute_quality_flags(summary, missing_df, df=df)
    
    assert flags['has_constant_columns'] is True
    assert 'const' in flags['constant_columns_list']
    
    assert flags['has_many_zero_values'] is True
    assert 'zeros' in flags['cols_with_many_zeros']
    
    assert flags['has_suspicious_id_duplicates'] is True