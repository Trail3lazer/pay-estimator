import pandas as pd
from data import DataManager
 
tests = [
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
]

for test in tests:
    try:
        test_input = pd.Series(test)

        result = DataManager()._clean_state(test_input)['state']
        assert result == test.get('expected')
    except Exception as e:
        print(test, result or 'No match')
        print(e)
