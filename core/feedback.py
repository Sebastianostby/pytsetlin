
from numba import njit
import numpy as np


@njit
def evaluate_clauses_training(literals, cb, n_literals):

    # captures the imply opperation ta -> lit?

    clause_outputs = np.ones(cb.shape[0], dtype=np.uint8)
    
    for clause_k in range(cb.shape[0]):

        for literal_k in range(n_literals):

            if(cb[clause_k][literal_k] > 0):

                if(literals[literal_k] == 0):

                    clause_outputs[clause_k] = 0

                    break
            
            if(cb[clause_k][literal_k + n_literals] > 0):

                if(literals[literal_k] == 1):   

                    clause_outputs[clause_k] = 0

                    break

    return clause_outputs


@njit
def evaluate_clause(literals, clause_block, n_literals):

    clause_outputs = np.ones(clause_block.shape[0], dtype=np.uint8)

    for clause_k in range(clause_block.shape[0]):

        is_empty_clause = True

        for literal_k in range(n_literals):

            if(clause_block[clause_k][literal_k] > 0):

                is_empty_clause = False

                if(literals[literal_k] == 0):

                    clause_outputs[clause_k] = 0

                    break
            
            if(clause_block[clause_k][literal_k + n_literals] > 0):

                is_empty_clause = False

                if(literals[literal_k] == 1):    

                    clause_outputs[clause_k] = 0

                    break

        if(is_empty_clause):

            clause_outputs[clause_k] = 0


    return clause_outputs


@njit
def get_update_p(wb, clause_outputs, threshold, y, target_class):
  
    vote_values = np.dot(wb.astype(np.float32), clause_outputs.astype(np.float32))

    vote_value = np.clip(np.array(vote_values[y]), -threshold, threshold)

    if target_class:
        return (threshold - vote_value) / (2 * threshold)
    else:
        return (threshold + vote_value) / (2 * threshold)



@njit
def update_clauses(cb, wb, clause_outputs, positive_prob, negative_prob, target, not_target, literals, n_literals, s):
    
    for clause_k in range(cb.shape[0]):

        if np.random.random() <= positive_prob:
            update_clause(cb[clause_k], wb[target], 1, literals, n_literals, clause_outputs[clause_k], clause_k, s) 

        if np.random.random() <= negative_prob:
            update_clause(cb[clause_k], wb[not_target], -1, literals, n_literals, clause_outputs[clause_k], clause_k, s) 
 



@njit
def update_clause(clause_row, clause_weight, target, literals, n_literals, clause_output, clause_k, s):
    
    sign = 1 if (clause_weight[clause_k] >= 0) else -1

    if(target*sign > 0):

        if(clause_output == 1):

            clause_weight[clause_k] += sign

            T1aFeedback(clause_row, literals, n_literals, s)

        else:
            
            T1bFeedback(clause_row, n_literals, s)

    elif(target*sign < 0):

        if clause_output == 1:

            clause_weight[clause_k] -= sign

            T2Feedback(clause_row, literals, n_literals)


@njit
def T1aFeedback(clause_row, literals, n_literals, s):
    s_inv = (1.0 / s)
    s_min1_inv = (s - 1.0) / s

    upper_state =  127
    lower_state = -127

    for literal_k in range(n_literals):

        if(literals[literal_k] == 1):

            if(np.random.random() <= s_min1_inv):

                if(clause_row[literal_k] < upper_state):

                    clause_row[literal_k] += 1


            if(np.random.random() <= s_inv):
                if(clause_row[literal_k + n_literals] > lower_state):
                    clause_row[literal_k + n_literals] -= 1          
            

        else:
            if(np.random.random() <= s_inv):
                if(clause_row[literal_k] > lower_state):
                    clause_row[literal_k] -= 1

            if(np.random.random() <= s_min1_inv):
                if(clause_row[literal_k + n_literals] < upper_state):
                    clause_row[literal_k + n_literals] += 1            

@njit
def T1bFeedback(clause_row, n_literals, s):

    s_inv = (1.0 / s)
    lower_state = -127

    
    for literal_k in range(n_literals):

        if np.random.random() <= s_inv:

            if clause_row[literal_k] > lower_state:
                clause_row[literal_k] -= 1

        if np.random.random() <= s_inv:

            if clause_row[literal_k + n_literals] > lower_state:
                clause_row[literal_k + n_literals] -= 1

@njit
def T2Feedback(clause_row, literals, n_literals):

    for literal_k in range(n_literals):

        if(literals[literal_k] == 0):
            if(clause_row[literal_k] <= 0):
                clause_row[literal_k] += 1

        else:
            if(clause_row[literal_k + n_literals] <= 0):
                
                clause_row[literal_k + n_literals] += 1