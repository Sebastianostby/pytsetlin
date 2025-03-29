import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytsetlin.data.mnist import get_mnist
from pytsetlin import TsetlinMachine




if __name__ == "__main__":

    X_train, X_test, y_train, y_test = get_mnist()

    tm = TsetlinMachine(n_clauses=500,
                        threshold=625,
                        s=10.0,
                        n_threads=20)

    tm.set_train_data(X_train, y_train)

    tm.set_eval_data(X_test, y_test)

    r = tm.train()
    
    print(r)


