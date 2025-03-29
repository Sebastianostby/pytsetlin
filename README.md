# PyTsetlin

A low-code, feature-POOR, Pythonic implementation of a Coalesced Tsetlin Machine. This is not intended to be a feature-rich or speed-optimized implementation; see relevant repositories like [
TMU](https://github.com/cair/tmu) and [green-tsetlin](https://github.com/ooki/green_tsetlin) for that. However, it's intended to be an easy-to-use TM programmed in Python, with the intent of making it accessible to plug-and-play new ideas and be able to get some results, either on an input level or TM memory level. Also, since the implementation is written entirely in Python, the code can be compared with the theoretical concepts presented in the papers, potentially making it easier to grasp.

Even though this repo is not focused on speed, I have made some functions compatible for Numba compilation. Without this, the code would be so slow that it deems the implementation unusable.

## Installation

1. Install package to environment to use in other projects:
```bash
pip install pytsetlin
```

2. Clone or template this repository and install the required dependencies:

```bash
cd pytsetlin
pip install -r requirements.txt
```

## Examples

### Basic training example

Here's a basic example of how to use the Tsetlin Machine:

```python
>>> from pytsetlin import TsetlinMachine
>>> from pytsetlin.data.mnist import get_mnist

>>> X_train, X_test, y_train, y_test = get_mnist()

>>> tm = TsetlinMachine(n_clauses=500,
                        threshold=625,
                        s=10.0,
                        n_threads=20)

>>> tm.set_train_data(X_train, y_train)

>>> tm.set_eval_data(X_test, y_test)

>>> r = tm.train(training_epochs=10)

# progress bar for visualization
Train Acc: 95.78, Eval Acc: 96.22, Best Eval Acc: 96.22 (10): 100%|███████████| 10/10 [00:55<00:00,  5.60s/it]

>>> print(r)
{'train_time': [10.82, 5.95, 5.08, 4.9, 4.65, 4.58, 4.44, 4.38, 4.35, 4.25], 'train_acc': [86.81, 92.18, 93.47, 94.04, 94.54, 94.91, 95.22, 95.53, 95.58, 95.78], 'eval_acc': [91.06, 93.01, 93.62, 94.3, 94.44, 94.73, 94.82, 94.97, 95.22, 96.22], 'best_eval_acc': 96.22, 'best_eval_epoch': 10}
```
Note performance may vary depending on system! 


### Investigating TM structure
Since the code is Pythonic, the TM structure can easily be investigated from the TsetlinMachine object:
```python
>>> # xor gate
>>> x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

>>> y = np.array([0, 1, 1, 0])

>>> tm = TsetlinMachine(n_clauses=4)

>>> tm.set_train_data(x, y)

>>> tm.train()

>>> print(tm.C) # get clause matrix
[[-35  25  24 -30]
 [-33 -41  12  23]
 [ 18 -38 -34  16]
 [ 17  15 -33 -42]]

>>> print(tm.W) # get weight matrix
[[-19  17 -20  16]
 [ 18 -19  18 -18]]
```

### Saving and loading

Any TM state can easly be saved during of after training

```python
>>> from pytsetlin import TsetlinMachine
>>> from pytsetlin.data.imdb import get_imdb

>>> X_train, X_test, y_train, y_test = get_imdb()

>>> tm = TsetlinMachine(n_clauses=500,
                        threshold=625,
                        s=2.0)

>>> tm.set_train_data(X_train, y_train)

>>> tm.set_eval_data(X_test, y_test)

>>> r = tm.train(training_epochs=10, save_best_state=True) # save during training

>>> tm.save_state(file_name='tm_state.npz') # save after training
```

Then saved memory, or any memory, can be used for predictions after: 

```python
>>> tm = TsetlinMachine()

>>> state = np.load('tm.state.npz')

>>> C = state['C'] # load clause matrix
>>> W = state['W'] # load weight matrix 


>>> clause_outputs = tm.evaluate_clauses(instance, memory=C) # what clauses matched the input
[0, 1, 0, 0, 1]

>>> class_sums = np.dot(W, clause_outputs) # majority voting
[-32, 55]

>>> prediction = np.argmax(class_sums)
1
```



## Literature References

* Core Papers 

     * [The Tsetlin Machine](https://arxiv.org/abs/1804.01508) introduces the Tsetlin Machine.

     * [Coalesced Multi-Output Tsetlin Machines](https://arxiv.org/abs/2108.07594) the variation this repo is largly based on.

* Other Cool Developments 

    * [Sparse Tsetlin Machine](https://arxiv.org/abs/2405.02375)


## Notes

1. Input data must be binary (dtype=np.uint8 for features, np.uint32 for labels)
2. The implementation uses Numba for efficient computation
3. Memory is allocated automatically when training begins


## License

MIT Licence

