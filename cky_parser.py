### NLU Assignment 3: PCFG and CKY Parser.
### Author: Aadesh Magare.
### 

#%%
from nltk.tree import Tree, ProbabilisticTree
from functools import reduce

#%%
#References: https://www.nltk.org/_modules/nltk/parse/viterbi.html
class Parser():
    def __init__(self, grammar):
        self.grammar = grammar
        self.unk = '<UNK>'

    def parse(self, tokens):
        try:
            self.grammar.check_coverage(tokens)
        except ValueError as v:
            # print('Words not Found', v)
            words = v.args[0].split(':')[1].replace('"', '').replace("'", "")[:-1]
            for word in words.split(','):
                w = word.strip()
                if w in tokens:
                    idx = tokens.index(w)
                    tokens[idx] = self.unk

        parse_table = {}

        for index in range(len(tokens)):
            token = tokens[index]
            parse_table[index, index + 1, token] = token

        for length in range(1, len(tokens) + 1):
            for start in range(len(tokens) - length + 1):
                span = (start, start + length)

                changed = True
                while changed:
                    changed = False

                    span_coverage = []
                    
                    for production in self.grammar.productions():
                        matching_rules = self.find_matching_rules(production.rhs(), span, parse_table)

                        for matching_rule in matching_rules:
                            span_coverage.append((production, matching_rule))

                    for (production, children) in span_coverage:
                        subtrees = [c for c in children if isinstance(c, Tree)]
                        p = reduce(lambda pr, t: pr * t.prob(), subtrees, production.prob())
                        node = production.lhs().symbol()
                        tree = ProbabilisticTree(node, children, prob=p)

                        c = parse_table.get((span[0], span[1], production.lhs()))
                        
                        if c is None or c.prob() < tree.prob():
                            parse_table[span[0], span[1], production.lhs()] = tree
                            changed = True

        tree = parse_table.get((0, len(tokens), self.grammar.start()))
       
        # if tree is None:
        #     [print(p, parse_table[p]) for p in parse_table if p[0] == 0 and p[1] == len(tokens)]
            # [print(p) for p in self.grammar.productions() if p.lhs() == Nonterminal('S')]

        return tree

    def find_matching_rules(self, rhs, span, parse_table):
        (start, end) = span

        if start >= end and rhs == ():
            return [[]]
        if start >= end or rhs == ():
            return []

        matching_rules = []
        for split in range(start, end + 1):
            l = parse_table.get((start, split, rhs[0]))
            if l is not None:
                rights = self.find_matching_rules(rhs[1:], (split, end), parse_table)
                matching_rules += [[l] + r for r in rights]

        return matching_rules

#%%
if __name__ == '__main__':
    from pcfg_grammar import load_grammar
    # grammar = load_grammar()
    print('Grammar Loaded')
    p = Parser(grammar)
    s = ['Pierre', 'Vinken', 'board']
    tree = p.parse(s)
    print(tree)