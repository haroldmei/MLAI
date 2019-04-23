
def insertVowelsGreedy(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    problem = VowelInsertionProblem(queryWords, bigramCost, possibleFills)
    state = (wordsegUtil.SENTENCE_BEGIN, 0)
    words = []
    while not problem.isEnd(state):
        frontier = util.PriorityQueue()
        for action, newState, cost in problem.succAndCost(state):
            frontier.update(newState, cost)
        s,p = frontier.removeMin()
        #print s[0], p
        words.append(s[0])
        state = s #(s[0], s[1]+1)

    return ' '.join(words)
    # END_YOUR_CODE


def segmentWordsGreedy(query, unigramCost):
    if len(query) == 0:
        return ''

    problem = SegmentationProblem(query, unigramCost)
    state = query
    words = []
    while not problem.isEnd(state):
        frontier = util.PriorityQueue()
        for action, newState, cost in problem.succAndCost(state):
            frontier.update(newState, cost)
        
        s,p = frontier.removeMin()
        #print state[:len(state)-len(s)], p
        words.append(state[:len(state)-len(s)])
        state = state[len(state)-len(s):]
    # BEGIN_YOUR_CODE (our solution is 10 lines of code, but don't worry if you deviate from this)    
    return ' '.join(words)
    # END_YOUR_CODE