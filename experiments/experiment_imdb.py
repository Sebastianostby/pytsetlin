import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytsetlin.data.imdb import get_imdb
from pytsetlin import TsetlinMachine




if __name__ == "__main__":

    x_train, y_train, x_test, y_test = get_imdb()

    tm = TsetlinMachine(n_clauses=500,
                        threshold=2000,
                        s=2.0)

    tm.set_train_data(x_train, y_train)

    tm.set_eval_data(x_test, y_test)

    r = tm.train(training_epochs=10)
    
    print(r)


