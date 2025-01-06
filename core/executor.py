

from numba import njit
import numpy as np


from core.feedback import evaluate_clauses_training, get_update_p, update_clauses, evaluate_clause



@njit
def train_epoch(cb, wb, x, y, threshold, s, n_outputs, n_literals):

    cb = np.ascontiguousarray(cb)
    wb = np.ascontiguousarray(wb)

    indices = np.arange(x.shape[0], dtype=np.int32) # this does not need to be calulated at every epoch...
    np.random.shuffle(indices)

    for indice in indices:
        
        literals = x[indice]
        target = y[indice]

        clause_outputs = evaluate_clauses_training(literals, cb, n_literals)

        update_ps = np.zeros(n_outputs, dtype=np.float32)

        for i in range(n_outputs):

            if i == target:
                update_ps[i] = get_update_p(wb, clause_outputs, threshold, i, True)
            else:
                update_ps[i] = get_update_p(wb, clause_outputs, threshold, i, False)    

        if np.sum(update_ps) == 0.0:
            return cb, wb

        
        not_target = np.random.randint(n_outputs)

        while not_target == target:
            not_target = np.random.randint(n_outputs)

        pos_update_p = update_ps[target]
        neg_update_p = update_ps[not_target]

        update_clauses(cb, wb, clause_outputs, pos_update_p, neg_update_p, target, not_target, literals, n_literals, s)

    return cb, wb


@njit
def classify(x, clause_block, weight_block, threshold, n_literals, n_outputs):

    clause_outputs = evaluate_clause(x, clause_block, n_literals)

    class_sums = np.dot(weight_block.astype(np.float32), clause_outputs.astype(np.float32))

    return np.argmax(class_sums)
  
@njit
def eval_predict(x_eval, cb, wb, threshold, n_outputs, n_literals):
    

    y_pred = np.zeros(x_eval.shape[0], dtype=np.int32)

    for i in range(x_eval.shape[0]):

        y_pred[i] = classify(x_eval[i], cb, wb, threshold, n_literals, n_outputs)


    return y_pred

