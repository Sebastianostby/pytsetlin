

from core import executor
import tqdm
import numpy as np



class TsetlinMachine:
    def __init__(self,
                 n_clauses:int = 50,
                 s:float = 5.0,
                 threshold:int = 100,
                 n_literals = None):

        self.n_clauses = n_clauses        
        self.s = s    
        self.threshold = threshold    

        self.n_literals = n_literals
        self.n_outputs = None

        self.C = None        
        self.W = None

        self.x_train = None
        self.y_train = None

        self.x_eval = None
        self.y_eval = None



    def set_train_data(self, instances:np.array, targets:np.array):
        

        if not isinstance(instances, np.ndarray):
            raise ValueError("x_train must be of type np.ndarray, x_train type: {}".format(type(instances)))

        if instances.shape[0] != targets.shape[0]:
            raise ValueError("Data x_train and y_train must have the same number of examples: {} != {}".format(instances.shape[0], targets.shape[0]))

        if instances.dtype != np.uint8:
            raise ValueError("Data x_train must be of type np.uint8, was: {}".format(instances.dtype))

        if targets.dtype != np.uint32:
            raise ValueError("Data y_train must be of type np.uint32, was: {}".format(targets.dtype))
        

        self.x_train = instances
        self.y_train = targets


        if self.n_literals is None:
            self.n_literals = self.x_train.shape[1] // 2

        self.n_outputs = len(np.unique(self.y_train)) 


    def set_eval_data(self, instances, targets):
        
        if not isinstance(instances, np.ndarray):
            raise ValueError("x_train must be of type np.ndarray, x_train type: {}".format(type(instances)))

        if instances.shape[0] != targets.shape[0]:
            raise ValueError("Data x_train and y_train must have the same number of examples: {} != {}".format(instances.shape[0], targets.shape[0]))

        if instances.dtype != np.uint8:
            raise ValueError("Data x_train must be of type np.uint8, was: {}".format(instances.dtype))

        if targets.dtype != np.uint32:
            raise ValueError("Data y_train must be of type np.uint32, was: {}".format(targets.dtype))


        self.x_eval = instances
        self.y_eval = targets

    
    def allocate_memory(self):


        if (self.n_literals is None) or (self.n_outputs is None):
            raise ValueError("failed to allocate memory, make sure data is set using set_train_data() and set_eval_data()")


        self.C = np.zeros((self.n_clauses, 2*self.n_literals), dtype=np.int8)

        self.W = np.random.choice(np.array([-1, 1]), size=(self.n_outputs, self.n_clauses), replace=True).astype(np.int32)


    def reset(self):
        pass


    def train(self, 
              training_epochs:int=10,
              eval_freq:int=1,
              hide_progress_bar:bool=False,
              early_stop_at:float=1.0):


        self.allocate_memory()



        with tqdm.tqdm(total=training_epochs, disable=hide_progress_bar) as progress_bar:
            progress_bar.set_description(f"[N/A/{training_epochs}], Train Acc: N/A, Eval Acc: N/A, Best Eval Acc: N/A")

            for epoch in range(training_epochs):

                self.C, self.W = executor.train_epoch(
                    self.C, self.W, self.x_train, self.y_train, 
                    self.threshold, self.s, self.n_outputs, self.n_literals
                    )

                if (epoch+1) % eval_freq == 0:

                    y_hat = executor.eval_predict(self.x_eval, self.C, self.W, self.threshold, self.n_outputs, self.n_literals)

                    score = np.mean(y_hat == self.y_eval)

                    print(score)

                progress_bar.set_description(f"[{epoch+1}/{training_epochs}]: Train Acc: N/A, Eval Acc: {score}, Best Eval Acc: N/A") 
                progress_bar.update(1)

                if score >= early_stop_at:
                    break
    

    def predict():
        pass




if __name__ == "__main__":

    tm = TsetlinMachine()

    tm.train()


