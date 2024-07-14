import pandas as pd
from data import DataManager
import pytest

class Test:
    @pytest.mark.parametrize('pri,sec,expected', [
        ((1,1,1),(None,None,None),(1,1,1)),
        ((1,1,1),(2,2,2),(1,1,1)),
        ((2,2,2),(1,1,1),(2,2,2)),
        ((0,1,0),(2,2,2),(1,2,1)),
        ((0,0,0),(1,1,1),(1,1,1)),
        ((None,None,None),(1,1,1),(1,1,1)),
    ])
    def test_should_substitute_columns_if_missing(self, pri, sec, expected):
        row = {
            'max_salary': pri[0],
            'med_salary': pri[1],
            'min_salary': pri[2],
            'max_salary0': sec[0],
            'med_salary0': sec[1],
            'min_salary0': sec[2],
        }
        row = DataManager().update_pay(row, 1)
        assert row['max_salary'] == expected[0]
        assert row['med_salary'] == expected[1]
        assert row['min_salary'] == expected[2]
        
    @pytest.mark.parametrize('pri,sec,mult,expected', [
        ((1,1,1),(None,None,None),3,(3,3,3)),
        ((1,1,1),(2,2,2),3,(3,3,3)),
        ((0,1,2),(2,2,2),3,(6,3,6)),
        ((0,0,0),(2,2,2),3,(6,6,6)),
        ((None,None,None),(2,2,2),3,(6,6,6)),
    ])
    def test_should_multipy_columns(self, pri, sec, mult, expected):
        row = {
            'max_salary': pri[0],
            'med_salary': pri[1],
            'min_salary': pri[2],
            'max_salary0': sec[0],
            'med_salary0': sec[1],
            'min_salary0': sec[2],
        }
        row = DataManager().update_pay(row, mult)
        assert row['max_salary'] == expected[0]
        assert row['med_salary'] == expected[1]
        assert row['min_salary'] == expected[2]
