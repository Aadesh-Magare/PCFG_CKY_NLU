### NLU Assignment 3: PCFG and CKY Parser.
### Author: Aadesh Magare.
### 

#%%
import nltk
from nltk.corpus import treebank
from nltk import tokenize
from nltk import Nonterminal, ProbabilisticProduction, Production
from nltk import PCFG, Tree
import pickle
from cky_parser import Parser
import time
from PYEVALB.scorer import Scorer
from PYEVALB import parser
import random, sys
seed = random.Random(77)

#%%
def read_split_treebank():
    fileids = treebank.fileids()
    seed.shuffle(fileids)
    split = int(len(fileids) * 0.8)
    x_train = fileids[:split]
    x_test = fileids[split:]
    print('Train - Test: ', len(x_train), len(x_test))
    return x_train, x_test
#%%
def create_grammar(x_train):
    productions = []
    for x in x_train:
        for tree in treebank.parsed_sents(x):
            # tree.collapse_unary(collapsePOS = True)
            tree.chomsky_normal_form()
            productions += tree.productions()

    S = Nonterminal('S')
    for w in ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP','WRB', 'NP' ]:
        productions.append(Production(Nonterminal(w), ('<UNK>', )))
    
    grammar = create_pcfg(S, productions)
    return grammar

#%%
def create_pcfg(start_symbol, productions):
    pcount = {}
    lcount = {}

    for prod in productions:
        lcount[prod.lhs()] = lcount.get(prod.lhs(), 0) + 1
        pcount[prod] = pcount.get(prod, 0) + 1

    prods = [
        ProbabilisticProduction(p.lhs(), p.rhs(), prob=pcount[p] / lcount[p.lhs()])
        for p in pcount
    ]

    # threshold= 5e-3
    # to_remove = [p for p in prods if p.is_lexical() and len(p) == 1 and p.prob() < threshold]
    
    # if to_remove:
    #     return create_pcfg(start_symbol, [p for p in prods if p.is_lexical() and len(p) == 1 and p.prob() > threshold])

    return PCFG(start_symbol, prods)

#%%
def save_grammar(grammar):
    
    with open('grammar.pickl', 'wb') as output_file:
        pickle.dump(grammar, output_file)
    
    # prods = {}
    # with open('grammar.txt', 'w') as output_file:
    #     for p in grammar.productions():
    #         if prods.get(p.lhs(), '') == '':
    #             prods[p.lhs()] = str(p).split(' -> ')[1]
    #         else:
    #             prods[p.lhs()] +=  ' | ' + str(p).split(' -> ')[1]
    #     for k, v in prods.items():
    #         output_file.write(f'{k} -> {v}\n')

#%%
def load_grammar():
    try:
        with open('grammar.pickl', 'rb') as input_file:
            grammar = pickle.load(input_file)
        
        # with open('grammar.txt', 'r') as input_file:
        #     grammar = PCFG.fromstring(input_file.read())
    except FileNotFoundError as NF:
        x_train, x_test = read_split_treebank()
        grammar = train_pcfg(x_train)
        save_grammar(grammar)

    return grammar

#%%
def train_pcfg(x_train):
    grammar = create_grammar(x_train)
    save_grammar(grammar)
    return grammar
    
#%%
def test_pcfg(x_test, grammar=None):
    if grammar is None:
        grammar = load_grammar()

    print('Grammar Ready')
    
    p = Parser(grammar)
    gold_test = []
    for idx, x in enumerate(x_test):
        for idx2, tree in enumerate(treebank.parsed_sents(x)):
            tree_words = list(map(lambda x: x.strip().replace('"', ''), tree.leaves()))
            print('Sentence:', tree_words)
            try:
                grammar.check_coverage(tree_words)
            except ValueError as v:
                
                new_words = set()
                words = v.args[0].split(':')[1].replace('"', '').replace("'", "")[:-1]
                for word in words.split(','):
                    new_words.add(word.strip())
                print('Treebank index', idx, idx2, 'OOV words:', new_words  , 'Applying Smoothing')
                # grammar = update_grammar(new_words, grammar, smoothing='Interpolation')
                grammar = update_grammar(new_words, grammar)
                p = Parser(grammar)
            try:
                test_tree = p.parse(tree_words)
                if test_tree :
                    print(test_tree)
                    gold_test.append((str(tree), str(test_tree.pformat())))
                    evaluate([gold_test[-1]])
                else:
                    print('No tree')
            except ValueError as v:
                print('No tree. ', v)
            
    print('Got parse trees for: ', len(gold_test), 'sentences.')
    return gold_test
#%%
def evaluate(gold_test):
    scorer = Scorer()
    for gold, test in gold_test:
        gold_tree = parser.create_from_bracket_string(gold)
        test_tree = parser.create_from_bracket_string(test)
        result = scorer.score_trees(gold_tree, test_tree)
        print(result)

#%%
def test(sentences):
    for sentence in sentences:
        tokens = sentence.split()
        grammar = load_grammar()
        p = Parser(grammar)
        tree = p.parse(tokens)
        if tree is None:
            print('No Tree')
        else:
            print(tree)
            tree.pretty_print()

#%%
def update_grammar(words, grammar, smoothing=None):
    # if smoothing is None use Add One.
    pcount = {}
    lcount = 0
    new_prods = []
    lhs = None
    for prod in grammar.productions():
        if str(prod.lhs()) == 'NN':
            lhs = prod.lhs()
            lcount += 1
            pcount[prod] = pcount.get(prod, 0) + 1

    add = len(words) + len(pcount)
    avg = 1 / lcount

    if lhs is None:
        lhs = Nonterminal('NN')

    for word in words:
        rhs = (word.strip("'"), )
        if smoothing is None:
            prob = 1 / (lcount + add)
        else:
            prob = avg / len(words)
        prod = ProbabilisticProduction(lhs, rhs, prob=prob)
        new_prods.append(prod)

    for p in grammar.productions():
        if str(p.lhs()) == 'NN':
            if smoothing is None:
                p = ProbabilisticProduction(p.lhs(), p.rhs(), prob= (pcount[p] + 1) / (lcount + add))
            else:
                p = ProbabilisticProduction(p.lhs(), p.rhs(), prob= p.prob() - (avg / lcount))
        new_prods.append(p)

    return PCFG(grammar.start(), new_prods)

#%%
def main(argv):
    if argv[0] == 'train':
        x_train, x_test = read_split_treebank()
        grammar = train_pcfg(x_train)
        print('Grammar ready')

    elif argv[0] == 'evaluate':
        x_train, x_test = read_split_treebank()
        gold_test = test_pcfg(x_test)
        evaluate(gold_test)

    elif argv[0] == 'test':
        test(argv[1:])

#%%
if __name__ == '__main__':
    main(sys.argv[1:])
#%%
