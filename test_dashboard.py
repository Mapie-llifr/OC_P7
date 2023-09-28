import pytest
from OCr_Projet_7 import dashboard_streamlit as dhdb



def test_scoring_accepted(mocker):
    mocker.patch("dhbd.predict", 
                 autospec = True,
                 return_value={'prediction': None, 
                               'score': 1, 
                               'id' : None})
    expected_value = {'body' : 'PRÊT ACCORDÉ', 'divider' : 'green'}
    assert dhdb.scoring() == expected_value


def test_scoring_risked(mocker):
    mocker.patch("dhbd.predict", 
                 autospec = True,
                 return_value={'prediction': None, 
                               'score': 5, 
                               'id' : None})
    expected_value = {'body' : 'PRÊT RISQUÉ', 'divider' : 'blue'}
    assert dhdb.scoring() == expected_value


def test_scoring_refused(mocker):
    mocker.patch("dhbd.predict", 
                 autospec = True,
                 return_value={'prediction': None, 
                               'score': 0, 
                               'id' : None})
    expected_value = {'body' : 'PRÊT REFUSÉ', 'divider' : 'red'}
    assert dhdb.scoring() == expected_value


def test_scoring_error(mocker):
    mocker.patch("dhbd.predict", 
                 autospec = True,
                 return_value={'prediction': None, 
                               'score': 8, 
                               'id' : None})
    expected_value = {'body' : 'erreur de calcul', 'divider' : 'grey'}
    assert dhdb.scoring() == expected_value