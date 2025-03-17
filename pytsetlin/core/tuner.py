import numpy as np
import optuna

def objective(trial, tm, training_epochs):

    tm.allocate_memory()

    tm.threshold = trial.suggest_int('threshold', 500, 20000, step=500)

    tm.s = trial.suggest_float('s', 1.0, 30.0, step=0.5)

    tm.boost_true_positives = trial.suggest_categorical('boost_true_positives', [True, False])
    
    do_literal_budget = trial.suggest_categorical('do_literal_budget', [True, False])


    if do_literal_budget:

        tm.n_literal_budget = trial.suggest_int('n_literal_budget', 10, 100, step=10)

    else:

        tm.n_literal_budget = np.inf    


    try:

        results = tm.train(hide_progress_bar=True, training_epochs=training_epochs)

    except:

        trial.report(0.0, step=0) 
        raise optuna.TrialPruned()


    return results['best_eval_acc']
