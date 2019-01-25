import sys
import pytest
sys.path.append('../../../batchflow')
from batchflow import Config

@pytest.mark.parametrize('variation', [0,1,2])
def test_dict_init(variation):

    processed_inputs = [{'a' : 1, 'b' : 2, 'a' : 3, 'b' : 4},
                        {'a' : 1, 'b/a' : 2, 'b' : 3, 'a/b' : 4},
                        {'a' : {'b' : 1}, 'a' : 2}]
                
    expected_inputs = [{'a': 3, 'b' : 4},
                       {'a/b' : 4, 'b' : 3},
                       {'a' : 2}]
    
    processed_config = Config(processed_inputs[variation])
    expected_config = Config(expected_inputs[variation])
                             
    assert expected_config.config == processed_config.config

@pytest.mark.parametrize('variation', [0,1])
def test_list_init(variation):

    processed_inputs = [[('a',1),('b',2),('a',3),('b',4)],
                        [('a', 1), ('b/a', 2), ('b', 3), ('a/b', 4)]]
                
    expected_inputs = [{'a': 3, 'b' : 4},
                       {'a/b' : 4, 'b' : 3}]
    
    processed_config = Config(processed_inputs[variation])
    expected_config = Config(expected_inputs[variation])
                             
    assert expected_config.config == processed_config.config

@pytest.mark.parametrize('variation', [0])
def test_config_init(variation):

    inputs = [Config({'a' : 0, 'b' : 1})]
    
    processed_config = Config(inputs[variation])
    expected_config = inputs[variation]
                             
    assert expected_config.config == processed_config.config
