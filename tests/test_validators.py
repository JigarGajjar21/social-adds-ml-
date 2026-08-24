"""Unit tests for input validation."""
import pytest
from src.validators import validate_prediction_input


class TestValidateInput:

    def test_valid_male_input(self):
        r = validate_prediction_input(1, 30, 50000)
        assert r.valid is True
        assert r.gender == 1
        assert r.age == 30
        assert r.salary == 50000

    def test_valid_female_input(self):
        r = validate_prediction_input(0, 25, 35000)
        assert r.valid is True
        assert r.gender == 0

    def test_string_inputs_are_coerced(self):
        r = validate_prediction_input("1", "30", "50000")
        assert r.valid is True
        assert r.age == 30

    def test_invalid_gender(self):
        r = validate_prediction_input(2, 30, 50000)
        assert r.valid is False
        assert "Gender" in r.error

    def test_age_too_low(self):
        r = validate_prediction_input(1, 0, 50000)
        assert r.valid is False
        assert "Age" in r.error

    def test_age_too_high(self):
        r = validate_prediction_input(1, 101, 50000)
        assert r.valid is False
        assert "Age" in r.error

    def test_negative_salary(self):
        r = validate_prediction_input(1, 30, -1000)
        assert r.valid is False
        assert "Salary" in r.error

    def test_unrealistic_salary(self):
        r = validate_prediction_input(1, 30, 20_000_000)
        assert r.valid is False
        assert "Salary" in r.error

    def test_non_numeric_input(self):
        r = validate_prediction_input("abc", "xyz", "foo")
        assert r.valid is False
        assert "numeric" in r.error.lower()

    def test_none_input(self):
        r = validate_prediction_input(None, None, None)
        assert r.valid is False

    def test_boundary_age_1(self):
        r = validate_prediction_input(1, 1, 0)
        assert r.valid is True

    def test_boundary_age_100(self):
        r = validate_prediction_input(0, 100, 1000000)
        assert r.valid is True
