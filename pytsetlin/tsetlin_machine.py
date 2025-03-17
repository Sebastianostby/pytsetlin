from pytsetlin.core import executor
from pytsetlin.core import config
from pytsetlin.core import tuner

from numba import set_num_threads
import tqdm
import numpy as np
import optuna

from time import perf_counter
import os

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')




class TsetlinMachine:
    def __init__(self,
                 n_clauses:int = 50,
                 s:float = 5.0,
                 threshold:int = 100,
                 n_literal_budget:int = np.inf,
                 boost_true_positives:bool = True, 
                 n_threads:int = 1):

        self.n_clauses = n_clauses        
        self.s = s    
        self.threshold = threshold    
        self.n_literal_budget = n_literal_budget
        self.boost_true_positives = boost_true_positives    

        self.C = None        
        self.W = None

        self.x_train = None
        self.y_train = None

        self.x_eval = None
        self.y_eval = None

        config.N_THREADS = n_threads
        config.OPERATE_PARALLEL = True if config.N_THREADS > 1 and not config.N_THREADS <= 0 else False
        set_num_threads(config.N_THREADS)


    def set_train_data(self, instances:np.array, targets:np.array):
        

        if not isinstance(instances, np.ndarray):
            raise ValueError("x_train must be of type np.ndarray, x_train type: {}".format(type(instances)))

        if instances.shape[0] != targets.shape[0]:
            raise ValueError("Data x_train and y_train must have the same number of examples: {} != {}".format(instances.shape[0], targets.shape[0]))

        if instances.dtype != np.uint8:
            raise ValueError("Data x_train must be of type np.uint8, was: {}".format(instances.dtype))

        if targets.dtype != np.uint32:
            raise ValueError("Data y_train must be of type np.uint32, was: {}".format(targets.dtype))
        

        self.n_literals = instances.shape[1]
        self.n_outputs = len(np.unique(targets)) 
        
        self.x_train = instances
        self.y_train = targets



    def set_eval_data(self, instances:np.array, targets:np.array):
        
        if not isinstance(instances, np.ndarray):
            raise ValueError("x_eval must be of type np.ndarray, x_eval type: {}".format(type(instances)))

        if instances.shape[0] != targets.shape[0]:
            raise ValueError("Data x_eval and y_eval must have the same number of examples: {} != {}".format(instances.shape[0], targets.shape[0]))

        if instances.dtype != np.uint8:
            raise ValueError("Data x_eval must be of type np.uint8, was: {}".format(instances.dtype))

        if targets.dtype != np.uint32:
            raise ValueError("Data y_eval must be of type np.uint32, was: {}".format(targets.dtype))
        

        self.x_eval = instances
        self.y_eval = targets

    
    def allocate_memory(self):


        if (self.n_literals is None) or (self.n_outputs is None):
            raise ValueError("failed to allocate memory, make sure data is set using set_train_data() and set_eval_data()")


        self.C = np.zeros((self.n_clauses, 2*self.n_literals), dtype=np.int8)
        self.W = np.random.choice(np.array([-1, 1]), size=(self.n_outputs, self.n_clauses), replace=True).astype(np.int32)


        if self.s < 1.0:
            raise ValueError('s must be larger or equal to 1.0. s >= 1.0')


        self.s_min_inv = (self.s - 1.0) / self.s
        self.s_inv = (1.0 / self.s)




    def reset(self):
        self.C = None        
        self.W = None

        self.x_train = None
        self.y_train = None

        self.x_eval = None
        self.y_eval = None


    def train(self, 
              training_epochs:int=10,
              eval_freq:int=1,
              hide_progress_bar:bool=False,
              early_stop_at:float=100.0,
              save_best_state=False,
              file_name="tm_state.npz",
              location_dir="saved_states"):


        full_path = os.path.join(location_dir, file_name)
        if os.path.exists(full_path):
            raise FileExistsError(f"File already exists: {full_path}")


        self.allocate_memory()

        r = {
            'train_time' : [],
            'train_acc' : [],
            'eval_acc' : [],
        }

        train_score = "N/A"
        eval_score = "N/A"
        best_eval_acc = "N/A"
        best_eval_epoch = "#"
        
        with tqdm.tqdm(total=training_epochs, disable=hide_progress_bar) as progress_bar:
            progress_bar.set_description(f"Train Acc: {train_score}, Eval Acc: {eval_score}, Best Eval Acc: {best_eval_acc} ({best_eval_epoch})")


            for epoch in range(training_epochs):

                st = perf_counter() 

                train_predictions = executor.train_epoch(
                                        self.C, self.W, self.x_train, self.y_train, 
                                        self.threshold, self.s_min_inv, self.s_inv, self.n_outputs, self.n_literals, self.n_literal_budget,
                                        self.boost_true_positives)
                                    
                et = perf_counter()
                
                if (epoch+1) % eval_freq == 0:

                    train_score = round(100 * np.mean(train_predictions == self.y_train).item(), 2)
                    r["train_acc"].append(train_score)

                    eval_predictions = executor.eval_predict(self.x_eval, self.C, self.W, self.n_literals)
                    eval_score = round(100 * np.mean(eval_predictions == self.y_eval).item(), 2)
                    r["eval_acc"].append(eval_score)

                    if best_eval_acc == 'N/A' or eval_score > best_eval_acc:
                        best_eval_acc = eval_score
                        best_eval_epoch = epoch+1
                        r["best_eval_acc"] = best_eval_acc
                        r["best_eval_epoch"] = best_eval_epoch


                        if save_best_state:
                            self.save_state(file_name=file_name, location_dir=location_dir)

                r["train_time"].append(round(et-st, 2))

                progress_bar.set_description(f"Train Acc: {train_score}, Eval Acc: {eval_score}, Best Eval Acc: {best_eval_acc} ({best_eval_epoch})") 
                progress_bar.update(1)

                if not eval_score == 'N/A':
                    if eval_score >= early_stop_at:
                        break

        return r
    

    def predict(self, x):
        
        return executor.classify(x, self.C, self.W, self.n_literals)


    def evaluate_clauses(self, literals, memory:np.array=None):

        if memory is None and self.C is None:
            raise ValueError('did not find loaded input memory or trained memory. either input a loaded memory or train tm.')

        if self.C is not None:
            memory = self.C

        return executor.evaluate_clause(literals, memory, literals.shape[0])



    def save_state(self, file_name = 'tm_state.npz', location_dir='saved_states'):

        if not os.path.isdir(f'{location_dir}'):
            os.mkdir(f'{location_dir}')

        np.savez(f"{location_dir}/{file_name}", C=self.C, W=self.W)


    def optimize_hyperparameters(self, n_trials=100, training_epochs=10):

        logging.info(f"This optimization only tunes threshold, s, boost_true_positives, and n_literal_budget. n_clauses is fixed at {self.n_clauses}.")

        self.allocate_memory()

        study = optuna.create_study(direction='maximize')

        study.optimize(lambda trial: tuner.objective(trial, self, training_epochs), n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params.copy()

        best_value = study.best_value
        
        if 'do_literal_budget' in best_params:
            if not best_params['do_literal_budget']:
                best_params['n_literal_budget'] = "infinity"
            del best_params['do_literal_budget']
        
        print(f"Score: {best_value:.4f}")
        print("Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")






if __name__ == "__main__":

    tm = TsetlinMachine(n_clauses=500)

    tm.optimize_hyperparameters()
