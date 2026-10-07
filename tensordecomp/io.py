from __future__ import annotations

import ast
import json
import re
from typing import Any

import numpy as np


def _to_clean_ndarray(data: Any) -> np.ndarray:
    try:
        arr = np.asarray(data, dtype=float)
    except ValueError as err:
        if "inhomogeneous" in str(err) or "setting an array element with a sequence" in str(err):
            raise ValueError("Incomplete or ragged tensor input: rows or sub-slices have mismatched lengths.") from err
        raise
    if arr.ndim == 0:
        raise ValueError("Scalar numbers are not multi-dimensional tensors. Provide a matrix or tensor array.")
    if arr.size == 0:
        raise ValueError("Tensor input contains no elements (empty tensor).")
    return arr


def parse_tensor_input(raw_value: str) -> np.ndarray:
    """Parse raw tensor input from JSON, Python literal, or text matrix formats into a NumPy ndarray.

    Args:
        raw_value: String representation of a tensor (JSON array, Python nested list,
                   bracketed notation, or whitespace/newline separated numbers).

    Returns:
        np.ndarray: Parsed float tensor.

    Raises:
        ValueError: If input is empty or cannot be parsed into a valid array.
    """
    text = (raw_value or "").strip()
    if not text:
        raise ValueError("Tensor input is empty")

    if text.count("[") != text.count("]"):
        raise ValueError("Incomplete tensor input: unclosed or unbalanced brackets detected.")

    # 1. Direct JSON parse
    try:
        data = json.loads(text)
        return _to_clean_ndarray(data)
    except json.JSONDecodeError:
        pass

    # 2. Direct Python literal_eval (handles trailing commas, tuples, Python numbers)
    try:
        data = ast.literal_eval(text)
        return _to_clean_ndarray(data)
    except (ValueError, SyntaxError):
        pass

    # 3. Extract bracketed tensor content [...]
    # Handles files/inputs containing shape headers or metadata above the tensor (e.g., "3\n2\n4\n1\n[[[[...]]]]")
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        bracket_content = text[first_bracket : last_bracket + 1].strip()

        # Try JSON on bracket content
        try:
            data = json.loads(bracket_content)
            return _to_clean_ndarray(data)
        except json.JSONDecodeError:
            pass

        # Try ast.literal_eval on bracket content
        try:
            data = ast.literal_eval(bracket_content)
            return _to_clean_ndarray(data)
        except (ValueError, SyntaxError):
            pass

        # Strip trailing commas before closing brackets: e.g. [1, 2,] -> [1, 2]
        cleaned_brackets = re.sub(r",\s*([\]\}])", r"\1", bracket_content)
        try:
            data = json.loads(cleaned_brackets)
            return _to_clean_ndarray(data)
        except json.JSONDecodeError:
            pass
        try:
            data = ast.literal_eval(cleaned_brackets)
            return _to_clean_ndarray(data)
        except (ValueError, SyntaxError):
            pass

        # Handle numpy-style bracketed formatting without commas: e.g. "[[ 1  2] [ 3  4]]"
        np_formatted = re.sub(r"(?<=[0-9eE\.\+\-])\s+(?=[0-9eE\.\+\-])", ", ", cleaned_brackets)
        np_formatted = re.sub(r"(?<=\])\s+(?=\[)", ", ", np_formatted)
        try:
            data = json.loads(np_formatted)
            return _to_clean_ndarray(data)
        except json.JSONDecodeError:
            pass
        try:
            data = ast.literal_eval(np_formatted)
            return _to_clean_ndarray(data)
        except (ValueError, SyntaxError):
            pass

    # 4. Dense flat numbers with potential shape headers
    tokens = text.replace(",", " ").replace("[", " ").replace("]", " ").split()
    if tokens:
        try:
            all_numbers = [float(t) for t in tokens]
        except ValueError as exc:
            raise ValueError(f"Invalid tensor format: {exc}") from exc

        # Check if text before first bracket provided explicit shape
        if first_bracket != -1:
            header_text = text[:first_bracket].strip()
            header_tokens = header_text.split()
            try:
                shape_dims = [int(t) for t in header_tokens]
                prod = int(np.prod(shape_dims))
                val_tokens = text[first_bracket:].replace(",", " ").replace("[", " ").replace("]", " ").split()
                vals = [float(t) for t in val_tokens]
                if len(vals) == prod:
                    return np.array(vals, dtype=float).reshape(shape_dims)
            except Exception:
                pass

        return np.array(all_numbers, dtype=float).reshape(-1, 1)

    raise ValueError("Could not parse tensor from the provided input.")
