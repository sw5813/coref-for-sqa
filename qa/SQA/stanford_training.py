import argparse, os, cPickle, sys, csv, glob, json, itertools, shutil
from unidecode import unidecode
from collections import Counter, defaultdict

def write_seq_parser_input(fold, out_name):
    
    reader = csv.DictReader(open(fold, 'r'), delimiter='\t')
    template = '(example (id %s) (utterance "%s") (context (graph tables.TableKnowledgeGraph %s)) '\
        + '(targetValue (list %s)))\n'
    mod_file = open('data/%s.examples' % out_name, 'w')

    for row in reader:
        qid = row['id']
        annotator = row['annotator']
        pos = row['position']
        question = row['question']
        table = row['table_file']
        answer = eval(row['answer_text'])
        qid = '%s_%s_%s' % (qid, annotator, pos)

        # only unique answers!
        ans_string = ''
        for ans in answer:
            ans = ans.strip().replace('"', '').replace('\\', '')
            ans_string += '(description "%s") ' % (ans)

        mod_file.write(template % (qid, question, table, ans_string.strip()))


if __name__ == '__main__':

    write_seq_parser_input('data/train.tsv', 'train')
    write_seq_parser_input('data/test.tsv', 'test')
    write_seq_parser_input('data/random-split-1-train.tsv', 'random-split-1-train')
    write_seq_parser_input('data/random-split-1-dev.tsv', 'random-split-1-dev')
    write_seq_parser_input('data/random-split-2-train.tsv', 'random-split-2-train')
    write_seq_parser_input('data/random-split-2-dev.tsv', 'random-split-2-dev')
    write_seq_parser_input('data/random-split-3-train.tsv', 'random-split-3-train')
    write_seq_parser_input('data/random-split-3-dev.tsv', 'random-split-3-dev')
    write_seq_parser_input('data/random-split-4-train.tsv', 'random-split-4-train')
    write_seq_parser_input('data/random-split-4-dev.tsv', 'random-split-4-dev')
    write_seq_parser_input('data/random-split-5-train.tsv', 'random-split-5-train')
    write_seq_parser_input('data/random-split-5-dev.tsv', 'random-split-5-dev')

    # execute this command in "acl2015" directory after unzipping their code to train:
    """
LC_ALL=C.UTF-8 java -ea -Dmodules=core,tables,corenlp -Xms8G -Xmx10G -cp libsempre/*:lib/* edu.stanford.nlp.sempre.Main -execDir state/execs/stanford_baseline.exec -overwriteExecDir -addToView 0 -jarFiles libsempre/* -executor tables.lambdadcs.LambdaDCSExecutor -Builder.valueEvaluator tables.TableValueEvaluator -TargetValuePreprocessor.targetValuePreprocessor tables.TableValuePreprocessor -NumberFn.unitless -NumberFn.alsoTestByConversion -TypeInference.typeLookup tables.TableTypeLookup -JoinFn.specializedTypeCheck false -JoinFn.typeInference true -Builder.parser FloatingParser -useSizeInsteadOfDepth -FloatingParser.maxDepth 15 -FloatingParser.useAnchorsOnce true -DerivationPruner.pruningStrategies singleton multipleSuperlatives sameMerge forwardBackward doubleNext emptyDenotation nonLambdaError badSuperlativeHead mistypedMerge -DerivationPruner.pruningComputers tables.TableDerivationPruningComputer -Grammar.inPaths grammars/combined.grammar -Grammar.tags alternative movement comparison superlative merge -Dataset.inPaths train,data/stanford_train.examples test,data/stanford_test.examples -Dataset.trainFrac 1 -Dataset.devFrac 0 -FeatureVector.ignoreZeroWeight -maxPrintedPredictions 10 -maxPrintedTrue 10 -logFeaturesLimit 10 -LanguageAnalyzer corenlp.CoreNLPAnalyzer -annotators tokenize ssplit pos lemma ner -combineFromFloatingParser -maxTrainIters 3 -showValues true -Params.l1Reg lazy -Params.l1RegCoeff 3e-5 -FeatureExtractor.featureComputers tables.features.PhrasePredicateFeatureComputer tables.features.PhraseDenotationFeatureComputer -usePredicateLemma -usePhraseLemmaOnly -maxNforLexicalizeAllPairs 2 -traverseWithFormulaTypes -reverseNameValueConversion allBang -lookUnderCellProperty -useGenericCellType -computeFuzzyMatchPredicates -Parser.beamsize 50 -featureDomains phrase-predicate phrase-denotation headword-denotation missing-predicate
"""
