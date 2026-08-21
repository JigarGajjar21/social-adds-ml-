from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    valid:   bool
    error:   Optional[str] = None
    gender:  Optional[int] = None
    age:     Optional[int] = None
    salary:  Optional[int] = None


def validate_prediction_input(gender, age, salary) -> ValidationResult:
    """
    Validate and coerce inputs for a prediction request.
    Accepts raw strings (from form) or ints (from JSON).
    """
    try:
        gender = int(gender)
        age    = int(age)
        salary = int(salary)
    except (ValueError, TypeError):
        return ValidationResult(valid=False, error="All fields must be numeric values.")

    if gender not in (0, 1):
        return ValidationResult(valid=False, error="Gender must be 0 (Female) or 1 (Male).")

    if not (1 <= age <= 100):
        return ValidationResult(valid=False, error="Age must be between 1 and 100.")

    if salary < 0:
        return ValidationResult(valid=False, error="Salary cannot be negative.")

    if salary > 10_000_000:
        return ValidationResult(valid=False, error="Salary value seems unrealistic (max $10,000,000).")

    return ValidationResult(valid=True, gender=gender, age=age, salary=salary)
