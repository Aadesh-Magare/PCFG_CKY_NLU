# PCFG and CKY Parser

# E1-246 2019: Assignment 3

**Usage:**
    
    python pcfg_grammar.py train        Start training model for generating PCFG, generated grammar is saved in 'grammar.pickl' in same directory.

    python pcfg_grammar.py evaluate     Grammar in 'grammar.pickl' is evaluated on test dataset, Precision, Recall and F1 score metrics are reported. 
                     
    python pcfg_grammar.py test 'your_sentence'     creates and prints a parse tree for given sentence using grammar saved in 'grammar.pickl'. 

**Trained Model:**

    Trained model (PCFG Grammar) can be downloaded from: https://drive.google.com/open?id=1ciamamWs6XDXhdCeGZ3tEBUtj59ypY13
    
**File Descriptions:**

    cky_parser.py       contains CKY parser implementation.
    pcfg_grammar.py     contains driver code and pcfg estimation.
    grammar.pickl       saved pcfg grammar.
     
**References:**

https://en.wikipedia.org/wiki/Probabilistic_context-free_grammar

https://en.wikipedia.org/wiki/Chomsky_normal_form

https://www.nltk.org/_modules/nltk/parse/viterbi.html

http://www.nltk.org/howto/corpus.html
