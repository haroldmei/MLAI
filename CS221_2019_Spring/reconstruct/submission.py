import shell
import util
import wordsegUtil

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        return self.query
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        return state == ''
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        result = []
        for i in range(1, len(state) + 1):
            result.append((i, state[i:], self.unigramCost(state[:i])))
        return result
        # END_YOUR_CODE

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=3)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 10 lines of code, but don't worry if you deviate from this)
    words = []
    for i in ucs.actions:
        words.append(query[:i])
        query = query[i:]
    
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
        print state[:len(state)-len(s)], p
        words.append(state[:len(state)-len(s)])
        state = state[len(state)-len(s):]
    # BEGIN_YOUR_CODE (our solution is 10 lines of code, but don't worry if you deviate from this)    
    return ' '.join(words)
    # END_YOUR_CODE

############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (wordsegUtil.SENTENCE_BEGIN, 0) #' '.join(self.queryWords))
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        return state[1] == len(self.queryWords)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 16 lines of code, but don't worry if you deviate from this)
        result = []
        current_token = self.queryWords[state[1]]
        next_state = state[1] + 1
        possibilities = self.possibleFills(current_token)
        if len(possibilities) == 0:
            possibilities.add(current_token)

        for i in possibilities:
            cost = self.bigramCost(state[0], i)
            result.append((i, (i, next_state), cost))
        return result
        # END_YOUR_CODE

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=3)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    words = []
    for i in ucs.actions: words.append(i)
    return ' '.join(words)
    # END_YOUR_CODE

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
        print s[0], p
        words.append(s[0])
        state = s #(s[0], s[1]+1)

    return ' '.join(words)
    # END_YOUR_CODE

############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        return (wordsegUtil.SENTENCE_BEGIN, self.query)
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        return state[1] == ''
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 23 lines of code, but don't worry if you deviate from this)
        result = []
        for index in range(1, len(state[1])+1):
            next_state = state[1][index:]
            current_token = state[1][:index]
            possibilities = self.possibleFills(current_token)
            for i in possibilities:
                cost = self.bigramCost(state[0], i)
                result.append((i, (i, next_state), cost))
        return result
        # END_YOUR_CODE

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 11 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=1)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    words = []
    for i in ucs.actions: words.append(i)
    return ' '.join(words)
    # END_YOUR_CODE

############################################################

if __name__ == '__main__':
    #shell.main()

    corpus = '.\leo-will.txt'
    unigramCost, bigramCost = wordsegUtil.makeLanguageModels(corpus)
    #print segmentWordsGreedy("whatisyourname", unigramCost)
    
    #line = wordsegUtil.cleanLine('mounted their horses and rode on')
    possibleFills = wordsegUtil.makeInverseRemovalDictionary(corpus, 'aeiou')
    print insertVowelsGreedy(['ths', 'ppl', 'wrkd', 'hpply'], bigramCost, possibleFills)
    print insertVowels(['ths', 'ppl', 'wrkd', 'hpply'], bigramCost, possibleFills)


    #smoothCost = wordsegUtil.smoothUnigramAndBigram(unigramCost, bigramCost, 0.2)
    #segmentAndInsert('mntdthrhrssndrdn', smoothCost, possibleFills)
#
    #parts = [wordsegUtil.removeAll(w, 'aeiou') for w in wordsegUtil.words(line)]
#
    #possibleFills = wordsegUtil.makeInverseRemovalDictionary(corpus, 'aeiou')
    #print '  ' + ' '.join(
    #    segmentAndInsert(part, smoothCost, possibleFills)
    #    for part in parts
    #)
