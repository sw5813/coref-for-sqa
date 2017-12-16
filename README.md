CPSC 490 Computer Science Senior Project  
=============================

Summer Wu, advised by Professor Dragomir Radev at Yale University

Basis of the code in directories: 
----------------
"qa"-  Pasupat and Liang's Floating Parser code, designed to answer questions based on the WikiTableQuestions dataset (https://worksheets.codalab.org/worksheets/0xf26cd79d4d734287868923ad1067cf4c/)

"e2e-coref"- Lee et al.'s End-to-End Coreference code (https://github.com/kentonl/e2e-coref)
	
Dataset:
---------
SQA: http://aka.ms/sqa  

Special thanks to Mohit Iyyer for providing me with code to apply FP to the SQA dataset, specifically, a script to convert .tsv files to .examples files

Experiment Workflow:
----------
(Assuming e2e-coref demo works, and FP code in qa works on WikiTableQuestions dataset)  

Activate python venv  

Apply coreference resolution to .tsv files in SQA:   
cd e2e-coref  
python replace_coref.py batched ../qa/SQA/data/  

Convert modified .tsv files in SQA to .examples files:  
cd ../qa/SQA/  
python stanford_training.py  

Modify the run.rb file for SQA:  
- Edit paths to data files in the "tableDataPaths" definition
- Derivation pruning: currently any complete logical form with > 10 items in its denotation get pruned. Remove this effect by commenting out the line "tooManyValues" in the run file.
- Features: there are some features that add bias on the denotation size. Disable the "custom-denotation" feature template altogether by replacing "@feat=all" in the command line with: ... @feat=some -featureDomains phrase-predicate phrase-denotation headword-denotation missing-predicate

Run FP over mod-train.tsv and mod-test.tsv files (and any other experiments!):  
./run.rb @cldir=1 @mode=tables @data=test @feat=some -featureDomains phrase-predicate phrase-denotation headword-denotation missing-predicate @train=1
