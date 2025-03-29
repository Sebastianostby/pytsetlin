import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
from pytsetlin.tsetlin_machine import TsetlinMachine
from pytsetlin.data import xor

@pytest.fixture
def sample_binary_data():

    X_train, y_train = xor.get_xor(n_rows = 1000, noise_fraction=0.2)
    X_test, y_test = xor.get_xor(n_rows = 200, noise_fraction=0.0)

    return X_train, y_train, X_test, y_test

@pytest.fixture
def basic_tsetlin():

    return TsetlinMachine(n_clauses=100, s=2.0, threshold=200, seed=12)

def test_initialization(basic_tsetlin):

    assert basic_tsetlin.n_clauses == 100
    assert basic_tsetlin.s == 2.0
    assert basic_tsetlin.threshold == 200
    assert basic_tsetlin.C is None
    assert basic_tsetlin.W is None

def test_data_validation(basic_tsetlin):

    invalid_x = np.random.rand(100, 10)  
    invalid_y = np.random.randint(2, size=50)  
    valid_x = np.random.randint(2, size=(100, 10), dtype=np.uint8)
    valid_y = np.random.randint(2, size=100, dtype=np.uint32)
    
    with pytest.raises(ValueError):
        basic_tsetlin.set_train_data(invalid_x, valid_y)
    
    with pytest.raises(ValueError):
        basic_tsetlin.set_train_data(valid_x, invalid_y)

def test_memory_allocation(basic_tsetlin, sample_binary_data):

    X_train, y_train, _, _ = sample_binary_data
        
    basic_tsetlin.set_train_data(X_train, y_train)
    basic_tsetlin.allocate_memory()
    
    expected_literal_count = X_train.shape[1] * 2  
    assert basic_tsetlin.C.shape == (basic_tsetlin.n_clauses, expected_literal_count)
    assert basic_tsetlin.W.shape == (basic_tsetlin.n_outputs, basic_tsetlin.n_clauses)


def test_training_performance(basic_tsetlin, sample_binary_data):

    X_train, y_train, X_test, y_test = sample_binary_data
    
    basic_tsetlin.set_train_data(X_train, y_train)
    basic_tsetlin.set_eval_data(X_test, y_test)
    
    results = basic_tsetlin.train(
        training_epochs=100,
        eval_freq=1,
        hide_progress_bar=True
    )
    
    assert 'train_time' in results
    assert 'eval_acc' in results
    

    predictions = np.zeros(X_test.shape[0])

    for i in range(X_test.shape[0]):

        predictions[i] = basic_tsetlin.predict(X_test[i])

    assert predictions.shape == y_test.shape
    assert np.all((predictions == 0) | (predictions == 1))  # binary predictions, this i can add as check to the set_ funcs
    
    accuracy = np.mean(predictions == y_test)
    assert accuracy == 1.0  



def test_singal_eval(basic_tsetlin):

    C = np.array([[100, -100], [-100, 100]])

    x = np.array([1])

    clause_outputs = basic_tsetlin.evaluate_clauses(x, memory=C)

    assert clause_outputs[0] == 1 
    assert clause_outputs[1] == 0

def test_custom_save_best_filename(basic_tsetlin, sample_binary_data):
    X_train, y_train, X_test, y_test = sample_binary_data
    
    basic_tsetlin.set_train_data(X_train, y_train)
    basic_tsetlin.set_eval_data(X_test, y_test)
    
    tmp_path = "saved_states"
    custom_filename = "my_custom_best_model.npz"
    custom_path = os.path.join(tmp_path, custom_filename)
    
    assert not os.path.exists(custom_path)
    
    results = basic_tsetlin.train(
        training_epochs=10,
        eval_freq=1,
        hide_progress_bar=True,
        save_best_state=True,
        file_name = custom_filename,
        location_dir = tmp_path
    )
    
    assert os.path.exists(custom_path)
    
    # Second training run - should raise FileExistsError
    with pytest.raises(FileExistsError):
        basic_tsetlin.train(
            training_epochs=10,
            eval_freq=1,
            hide_progress_bar=True,
            save_best_state=True,
            file_name= custom_filename,
            location_dir=tmp_path
        )

    if os.path.exists(custom_path):
        os.remove(custom_path)
