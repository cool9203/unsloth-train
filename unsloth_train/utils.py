# coding: utf-8

import re
from inspect import signature
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import pandas as pd

_latex_table_begin_pattern = r"\\begin{tabular}{[lrc|]*}"
_latex_table_end_pattern = r"\\end{tabular}"
_latex_multicolumn_pattern = r"\\multicolumn{(\d+)}{([lrc|]+)}{(.*)}"
_latex_multirow_pattern = r"\\multirow{(\d+)}{([\*\d]+)}{(.*)}"


def _get_function_used_params(
    callable: Callable,
    **kwds: Dict,
) -> Dict[str, Any]:
    """Get `callable` need parameters from kwds.

    Args:
        callable (Callable): function

    Returns:
        Dict[str, Any]: parameters
    """
    parameters = dict()
    callable_parameters = signature(callable).parameters
    for parameter, value in kwds.items():
        if parameter in callable_parameters:
            if value:
                parameters.update({parameter: value})
            else:
                parameters.update({parameter: None})
    return parameters


def preprocess_latex_table_string(
    latex_table_str: str,
) -> str:
    processed_latex_table_str = re.sub(_latex_table_begin_pattern, "", latex_table_str)
    processed_latex_table_str = re.sub(_latex_table_end_pattern, "", processed_latex_table_str)
    processed_latex_table_str = processed_latex_table_str.replace("\n", " ").strip()

    # Fix multiple \hline and \hline not at start of row error
    rows = processed_latex_table_str.split(r"\\")
    new_rows = list()
    for row in rows:
        _row = row
        if row.count(r"\hline") > 0:
            _row = _row.replace(r"\hline", "").strip()
            _row = rf"\hline {_row}"
        new_rows.append(_row)

    return "\\\\\n".join(new_rows)


def pre_check_latex_table_string(
    latex_table_str: str,
) -> Tuple[str, str]:
    results = re.findall(_latex_table_begin_pattern, latex_table_str)
    if not results:
        raise ValueError("Not latex table")
    elif len(results) > 1:
        raise ValueError("Not support convert have multi latex table")

    begin_str = results[0]
    end_str = r"\end{tabular}"
    return (begin_str, end_str)


def convert_latex_table_to_pandas(
    latex_table_str: str,
    headers: Union[bool, Sequence[str], None] = None,
    unsqueeze: bool = False,
) -> pd.DataFrame:
    pre_check_latex_table_string(latex_table_str=latex_table_str)
    processed_latex_table_str = preprocess_latex_table_string(latex_table_str)
    rows = [
        row.replace("\n", "").strip()
        for row in processed_latex_table_str.split(r"\\")
        if ("&" in row or r"\multicolumn" in row) and row.replace("\n", "").strip()
    ]  # Filter unrelated row data

    # Split latex table to list table
    cleaned_data = list()
    table_data = [row.replace(r"\\", "").replace(r"\hline", "").replace(r"\cline", "").strip().split("&") for row in rows]
    for row in table_data:
        _row_data = list()
        for cell in row:
            if re.match(_latex_multicolumn_pattern, cell):
                multicolumn_data = re.findall(_latex_multicolumn_pattern, cell)[0]
                for index in range(int(multicolumn_data[0])):
                    if unsqueeze:
                        _row_data.append(multicolumn_data[2].strip())
                    else:
                        if index == 0:
                            _row_data.append(
                                rf"\multicolumn{{{multicolumn_data[0]}}}{{{multicolumn_data[1]}}}{{{multicolumn_data[2].strip()}}}"
                            )
                        else:
                            _row_data.append("")
            else:
                _row_data.append(cell.strip())
        cleaned_data.append(_row_data)

    # Process multirow
    for col in range(len(cleaned_data)):
        for row in range(len(cleaned_data[col])):
            # Clean multi row data
            multirow_result = re.findall(_latex_multirow_pattern, cleaned_data[col][row])
            if multirow_result:
                if unsqueeze:
                    for offset in range(int(multirow_result[0][0])):
                        cleaned_data[col + offset][row] = multirow_result[0][2].strip()
                else:
                    cleaned_data[col][row] = (
                        rf"\multirow{{{multirow_result[0][0]}}}{{{multirow_result[0][1]}}}{{{multirow_result[0][2].strip()}}}"
                    )
                    for offset in range(1, int(multirow_result[0][0])):
                        cleaned_data[col + offset][row] = ""

    try:
        if headers:
            if isinstance(headers, bool):
                headers = cleaned_data[0]  # First row is header
                cleaned_data = cleaned_data[1:]  # Other row is row data

            # Filling every row length to headers length
            for i in range(len(cleaned_data)):
                if len(cleaned_data[i]) > len(headers):
                    cleaned_data[i] = cleaned_data[i][: len(headers)]
                elif len(cleaned_data[i]) < len(headers):
                    cleaned_data[i] += ["" for _ in range(len(headers) - len(cleaned_data[i]))]
            df = pd.DataFrame(cleaned_data, columns=headers)
        else:
            df = pd.DataFrame(cleaned_data)
    except ValueError as e:
        raise ValueError("Not support this latex") from e

    return df


def convert_pandas_to_latex(
    df: pd.DataFrame,
    full_border: bool = False,
) -> str:
    _row_before_text = ""
    if full_border:
        _row_before_text = r"\hline "
        latex_table_str = f"\\begin{{tabular}}{{{'c'.join(['|' for _ in range(len(df.columns) + 1)])}}}\n"
    else:
        latex_table_str = f"\\begin{{tabular}}{{{''.join(['c' for _ in range(len(df.columns))])}}}\n"

    # Add header
    latex_table_str += _row_before_text + f"{'&'.join([column for column in df.columns])}\\\\\n"

    # Add row data
    for i in range(len(df)):
        row = list()
        skip_count = 0
        for column in df.columns:
            if skip_count > 0:
                skip_count -= 1
            else:
                multicolumn_result = re.findall(_latex_multicolumn_pattern, df[column].iloc[i])
                skip_count = int(multicolumn_result[0][0]) - 1 if multicolumn_result and skip_count == 0 else skip_count
                row.append(df[column].iloc[i])
        latex_table_str += _row_before_text + f"{'&'.join(row)}\\\\\n"

    if full_border:
        latex_table_str += "\\hline\n"
    latex_table_str += r"\end{tabular}"

    return latex_table_str
