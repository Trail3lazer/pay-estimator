import pandas as pd
from data import DataManager
import pytest
import unittest 


class Test:
    
    @pytest.mark.parametrize('test',[
        {
            'location': 'Fort Collins',
            'state': 'Fort Collins, wA',
            'expected': 'WA'
        },
        {
            'location': 'Fort Collins, wA',
            'state': 'Fort Collins, Alaska',
            'expected': 'WA'
        },
        {
            'location': 'Fort Collins, Washington',
            'state': 'AK',
            'expected': 'WA'
        },
        {
            'location': 'Washington DC',
            'state': 'AK',
            'expected': 'DC'
        },
        {
            'location': None,
            'state': 'Fort Collins, wA',
            'expected': 'WA'
        },
        {
            'location': 'Fort Collins, wA',
            'state': None,
            'expected': 'WA'
        },
        {
            'location': 'Fort Collins',
            'state': None,
            'expected': None
        },
        {
            'location': None,
            'state': None,
            'expected': None
        },
    ])
    def test_should_match_states(self, test):
        result = 'No match'
        test_input = pd.Series(test)
        result = DataManager()._clean_state(test_input)['state']
        assert result == test.get('expected')
        
        
        
if __name__ == '__main__':
    pytest.main()