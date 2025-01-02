
# experiments_xor.py 

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.xor import get_xor


from tsetlin_macine import TsetlinMachine






if __name__ == "__main__":

    x, y = get_xor(flip_fraction=0.0)


    tm = TsetlinMachine(n_clauses=1000,
                        threshold=500,
                        s=2.0)

    tm.set_train_data(x, y)

    tm.set_eval_data(x, y)

    tm.train(training_epochs=1000)



