"""
File Validation Utilities for FibriNet.

Provides security checks for input files to prevent denial-of-service
attacks via excessively large files or malformed content.
"""

import os
from typing import Optional


class FileSizeError(ValueError):
    """Raised when a file exceeds the maximum allowed size."""
    pass


class FileValidationError(ValueError):
    """Raised when file validation fails."""
    pass


# Configuration constants
DEFAULT_MAX_FILE_SIZE_MB = 50  # 50 MB default limit
DEFAULT_MAX_FILE_SIZE_BYTES = DEFAULT_MAX_FILE_SIZE_MB * 1024 * 1024


def validate_file_size(
    file_path: str,
    max_size_bytes: Optional[int] = None,
    max_size_mb: Optional[float] = None
) -> int:
    """
    Validate that a file does not exceed the maximum allowed size.

    Args:
        file_path: Path to the file to validate
        max_size_bytes: Maximum allowed size in bytes (takes precedence)
        max_size_mb: Maximum allowed size in megabytes

    Returns:
        Actual file size in bytes

    Raises:
        FileNotFoundError: If file does not exist
        FileSizeError: If file exceeds maximum size
        FileValidationError: If file cannot be accessed
    """
    # Determine max size
    if max_size_bytes is not None:
        max_size = max_size_bytes
    elif max_size_mb is not None:
        max_size = int(max_size_mb * 1024 * 1024)
    else:
        max_size = DEFAULT_MAX_FILE_SIZE_BYTES

    # Check file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check it's a file, not a directory
    if not os.path.isfile(file_path):
        raise FileValidationError(f"Path is not a file: {file_path}")

    # Get file size
    try:
        file_size = os.path.getsize(file_path)
    except OSError as e:
        raise FileValidationError(f"Cannot access file {file_path}: {e}")

    # Validate size
    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        raise FileSizeError(
            f"File '{os.path.basename(file_path)}' is too large.\n"
            f"  Size: {actual_mb:.2f} MB\n"
            f"  Maximum allowed: {max_mb:.2f} MB\n"
            f"Please use a smaller file or contact support if you need to process larger datasets."
        )

    return file_size


def validate_file_extension(
    file_path: str,
    allowed_extensions: tuple[str, ...] = ('.xlsx', '.xls', '.csv')
) -> str:
    """
    Validate that a file has an allowed extension.

    Args:
        file_path: Path to the file to validate
        allowed_extensions: Tuple of allowed extensions (with dots)

    Returns:
        The file extension (lowercase, with dot)

    Raises:
        FileValidationError: If extension is not allowed
    """
    _, ext = os.path.splitext(file_path)
    ext_lower = ext.lower()

    if ext_lower not in allowed_extensions:
        raise FileValidationError(
            f"Unsupported file type: '{ext}'\n"
            f"Allowed types: {', '.join(allowed_extensions)}"
        )

    return ext_lower


def validate_input_file(
    file_path: str,
    max_size_mb: Optional[float] = None,
    allowed_extensions: tuple[str, ...] = ('.xlsx', '.xls', '.csv')
) -> dict:
    """
    Perform comprehensive validation on an input file.

    Args:
        file_path: Path to the file to validate
        max_size_mb: Maximum allowed size in megabytes
        allowed_extensions: Tuple of allowed extensions

    Returns:
        Dictionary with validation results:
        {
            'path': str,
            'size_bytes': int,
            'size_mb': float,
            'extension': str,
            'valid': True
        }

    Raises:
        FileNotFoundError: If file does not exist
        FileSizeError: If file exceeds maximum size
        FileValidationError: If validation fails
    """
    # Validate extension first (fast check)
    extension = validate_file_extension(file_path, allowed_extensions)

    # Validate size
    size_bytes = validate_file_size(
        file_path,
        max_size_mb=max_size_mb or DEFAULT_MAX_FILE_SIZE_MB
    )

    return {
        'path': file_path,
        'size_bytes': size_bytes,
        'size_mb': size_bytes / (1024 * 1024),
        'extension': extension,
        'valid': True
    }
