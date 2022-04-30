from collections import deque
from support import definite_clause

### THIS IS THE TEMPLATE FILE
### WARNING: DO NOT CHANGE THE NAME OF FILE OR THE FUNCTION SIGNATURE


def pl_fc_entails(symbols_list : list, KB_clauses : list, known_symbols : list, query : int) -> bool:
    """
    pl_fc_entails function executes the Propositional Logic forward chaining algorithm (AIMA pg 258).
    It verifies whether the Knowledge Base (KB) entails the query
        Inputs
        ---------
            symbols_list  - a list of symbol(s) (have to be integers) used for this inference problem
            KB_clauses    - a list of definite_clause(s) composed using the numbers present in symbols_list
            known_symbols - a list of symbol(s) from the symbols_list that are known to be true in the KB (facts)
            query         - a single symbol that needs to be inferred

            Note: Definitely check out the test below. It will clarify a lot of your questions.

        Outputs
        ---------
        return - boolean value indicating whether KB entails the query
    """

    # Initialize agenda queue, count array, and inferred dictionary
    # a queue of symbols, initially all symbols known to be true in knowledge base
    agenda = deque(known_symbols[:]) 

    # table, where count[c] is number of symbols in clause c's premise
    count = []
    for clause in KB_clauses:
        count.append(len(clause.body))
    
    # a table, where inferred[s] in initially false for all symbols
    inferred = dict()
    for symbol in symbols_list:
        inferred[symbol] = False

    # Forward Chaning algorithm, from AIMA pg 258
    # while agenda is not empty
    while agenda:
        # pop symbol from queue
        p = agenda.popleft()

        # if symbol is query, return True as we have found it
        if p == query:
            return True

        # if symbol is not already inferred
        if not inferred[p]:
            inferred[p] = True # update inferred

            # for each clause in knowledge base where p is in c.premise, decrement count[c] by 1 and if count[c] becomes 0, 
            # add it's conclusion to the agenda.
            for i in range(len(KB_clauses)):
                if p in KB_clauses[i].body:
                    count[i]-=1
                    if count[i]==0:
                        agenda.append(KB_clauses[i].conclusion)
    
    # since agenda empty, cannot continue forward search, so symbol cannot be concluded from KB (i.e., return false)
    return False

# SAMPLE TEST
if __name__ == '__main__':

    # Symbols used in this inference problem (Has to be Integers)
    symbols = [1,2,9,4,5]

    # Clause a: 1 and 2 => 9
    # Clause b: 9 and 4 => 5
    # Clause c: 1 => 4
    KB = [definite_clause([1, 2], 9), definite_clause([9,4], 5), definite_clause([1], 4)]

    # Known Symbols 1, 2
    known_symbols = [1, 2]

    # Does KB entail 5?
    entails = pl_fc_entails(symbols, KB, known_symbols, 5)

    print("Sample Test: " + ("Passed" if entails == True else "Failed"))
