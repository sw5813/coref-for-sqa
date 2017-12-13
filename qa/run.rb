#!/usr/bin/ruby

# This is the main entry point for running all SEMPRE programs.  See
# fig/lib/execrunner.rb for more documentation for how commands are generated.
# There are a bunch of modes that this script can be invoked with, which
# loosely correspond to the modules.

$: << 'fig/lib'
require 'execrunner'

$modes = []
def addMode(name, description, func)
  $modes << [name, description, func]
end

def codalab
  # Set @cl=1 to run job on CodaLab
  l(
    letDefault(:cl, 0),
    sel(:cl,
      l(),
      l('cl', 'run', ':fig', ':lib', ':module-classes.txt', ':libsempre', '---'),
    nil),
  nil)
end

def header(modules='core')
  l(
    codalab,
    # Queuing system
    letDefault(:q, 0), sel(:q, l(), l('fig/bin/q', '-shareWorkingPath', o('mem', '5g'), o('memGrace', 10), '-add', '---')),
    # Create execution directory
    'fig/bin/qcreate',
    # Run the Java command...
    'LC_ALL=C.UTF-8',
    'java',
    '-ea',
    '-Dmodules='+modules,
    # Memory size
    letDefault(:memsize, 'default'),
    sel(:memsize, {
      'tiny' => l('-Xms2G', '-Xmx4G'),
      'low' => l('-Xms5G', '-Xmx7G'),
      'default' => l('-Xms8G', '-Xmx10G'),
      'medium' => l('-Xms12G', '-Xmx14G'),
      'high' => l('-Xms20G', '-Xmx24G'),
      'higher' => l('-Xms40G', '-Xmx50G'),
      'impressive' => l('-Xms75G', '-Xmx90G'),
    }),
    # Classpath
    '-cp', 'libsempre/*:lib/*',
    # Profiling
    letDefault(:prof, 0), sel(:prof, l(), '-Xrunhprof:cpu=samples,depth=100,file=_OUTPATH_/java.hprof.txt'),
  nil)
end

def rlwrap; system('which rlwrap') ? 'rlwrap' : nil end

def unbalancedTrainDevSplit
  l(o('Dataset.trainFrac', 0.8), o('Dataset.devFrac', 0.2))
end
def balancedTrainDevSplit
  l(o('Dataset.trainFrac', 0.5), o('Dataset.devFrac', 0.5))
end

def figOpts; l(o('execDir', '_OUTPATH_'), o('overwriteExecDir'), o('addToView', 0)) end

############################################################
# Unit tests

addMode('test', 'Run unit tests', lambda { |e|
  l(
    'java', '-ea', '-Xmx12g', '-cp', 'libsempre/*:lib/*',
    lambda { |e|
      e.key?(:sparqlserver) ? "-Dsparqlserver=http://#{e[:sparqlserver]}/sparql" : l()
    },
    'org.testng.TestNG',
    lambda { |e|
      if e[:class]
        l('-testclass', 'edu.stanford.nlp.sempre.' + e[:class])
      else
        'testng.xml'
      end
    },
    lambda { |e|
      if e[:fast]
        o('excludegroups', 'sparql,corenlp')
      else
        nil
      end
    },
  nil)
})

############################################################
# Freebase

def freebaseHeader; header('core,freebase') end

def freebaseFeatureDomains
  [
    'basicStats',
    'alignmentScores',
    'entityFeatures',
    'context',
    'skipPos',
    'joinPos',
    'wordSim',
    'lexAlign',
    'tokenMatch',
    'rule',
    'opCount',
    'constant',
    'denotation',
    'whType',
    'span',
    'derivRank',
    'lemmaAndBinaries',
  nil].compact
end

def sparqlOpts
  l(
    required(:sparqlserver, 'host:port of the Sparql server'), # Example: jonsson:3093, etc.
    o('SparqlExecutor.endpointUrl', lambda{|e| 'http://'+e[:sparqlserver]+'/sparql'}),
  nil)
end

def freebaseOpts
  l(
    figOpts,
    sparqlOpts,

    # Features
    o('FeatureExtractor.featureDomains', *freebaseFeatureDomains),
    o('Builder.executor', 'freebase.SparqlExecutor'),
    o('Builder.valueEvaluator', 'freebase.FreebaseValueEvaluator'),
    o('LanguageAnalyzer.languageAnalyzer', 'corenlp.CoreNLPAnalyzer'),

    # Lexicon
    o('LexiconFn.lexiconClassName', 'edu.stanford.nlp.sempre.fbalignment.lexicons.Lexicon'),
    l( # binary
      o('BinaryLexicon.binaryLexiconFilesPath', 'lib/fb_data/7/binaryInfoStringAndAlignment.txt'),
      o('BinaryLexicon.keyToSortBy', 'Intersection_size_typed'),
    nil),
    o('UnaryLexicon.unaryLexiconFilePath', 'lib/fb_data/7/unaryInfoStringAndAlignment.txt'), # unary
    o('EntityLexicon.entityPopularityPath', 'lib/fb_data/7/entityPopularity.txt'), # entity
  nil)
end

def cachePaths(lexiconFnCachePath, sparqlExecutorCachePath)
  l(
    required(:cacheserver, 'none (don\'t cache to disk), local (write to local file), or <hostname>:<port> (hit the cacheserver)'),
    lambda { |e|
      cacheserver = e[:cacheserver]
      cacheserver = 'jonsson:4000' if cacheserver == 'remote' # Default
      case cacheserver
      when 'none' then l()
      when 'local' then l( # Use files directly - don't run more than one job that does this!
        o('Lexicon.cachePath', 'LexiconFn.cache'),
        o('SparqlExecutor.cachePath', 'SparqlExecutor.cache'),
        o('FreebaseSearch.cachePath', 'FreebaseSearch.cache'),
      nil)
      else l(
        o('Lexicon.cachePath', cacheserver+':/u/nlp/data/semparse/cache/'+lexiconFnCachePath),
        o('SparqlExecutor.cachePath', cacheserver+':/u/nlp/data/semparse/cache/'+sparqlExecutorCachePath),
        o('FreebaseSearch.cachePath', cacheserver+':/u/nlp/data/semparse/cache/fbsearch/1.cache'),
        # Read-only
        o('EntityLexicon.mid2idPath', cacheserver+':/u/nlp/data/semparse/scr/freebase/freebase-rdf-2013-06-09-00-00.canonical-id-map'),
        o('FreebaseTypeLookup.entityTypesPath', cacheserver+':/u/nlp/data/semparse/scr/freebase/freebase-rdf-2013-06-09-00-00.canonicalized.en-types'),
      nil)
      end
    },
  nil)
end

# tag is either "free917" or "webquestions"
def emnlp2013AblationExperiments(tag)
  l(
    letDefault(:ablation, 0),
    # Ablation experiments (EMNLP)
    sel(:ablation,
      l(), # (0) Just run things normally
      selo(nil, 'Parser.beamSize', 200, 50, 10), # (1) Vary beam size
      selo(nil, 'Dataset.trainFrac', 0.1, 0.2, 0.4, 0.6), # (2) Vary training set size
      sel(nil, # (3) Structural: only do join or only do bridge
        o('Grammar.tags', l(tag, 'join')),
        o('Grammar.tags', l(tag, 'bridge')),
        o('Grammar.tags', l(tag, 'inject')),
      nil),
      sel(nil, # (4) Features
        o('FeatureExtractor.featureDomains', *(freebaseFeatureDomains+['lexAlign'])), # +lexAlign
        o('FeatureExtractor.featureDomains', *(freebaseFeatureDomains+['lexAlign']-['alignmentScores'])), # +lexAlign -alignmentScores
        o('FeatureExtractor.featureDomains', *(freebaseFeatureDomains-['denotation'])), # -denotation
        o('FeatureExtractor.featureDomains', *(freebaseFeatureDomains-['skipPos', 'joinPos'])), # -syntax features (skipPos, joinPos)
      nil),
      #o('Builder.executor', 'FormulaMatchExecutor'), # (6) train on logical forms (doesn't really work well)
    nil),

    letDefault(:split, 0), selo(:split, 'Dataset.splitRandom', 1, 2, 3),
  nil)
end

def free917
  l( # Data
    letDefault(:data, 0),
    sel(:data,
      l(o('Dataset.inPaths', 'train,data/free917.train.examples.canonicalized.json'), unbalancedTrainDevSplit), # (0) train 0.8, dev 0.2
      l(o('Dataset.inPaths', 'train,data/free917.train.examples.canonicalized.json', 'test,data/free917.test.examples.canonicalized.json')), # (1) Don't run on test yet!
    nil),

    # Grammar
    o('Grammar.inPaths', 'freebase/data/emnlp2013.grammar'),
    o('Parser.beamSize', 500), 

    emnlp2013AblationExperiments('free917'),

    # lexicon index
    letDefault(:lucene, 0),
    sel(:lucene,
      l(
        o('EntityLexicon.exactMatchIndex','lib/lucene/4.4/free917/'),
        cachePaths('10/LexiconFn.cache', '10/SparqlExecutor.cache'),
        o('Grammar.tags', 'free917', 'bridge', 'join', 'inject', 'exact'),
      nil),
      l( # With entity disambiguation - currently too crappy
        o('EntityLexicon.inexactMatchIndex','lib/lucene/4.4/inexact/'),
        cachePaths('4/LexiconFn.cache', '4/SparqlExecutor.cache'),
        o('Grammar.tags', 'free917', 'bridge', 'join', 'inject', 'inexact'),
      nil),
    nil),
    # Use binary predicate features (overfits on free917)
    o('BridgeFn.filterBadDomain',false),
    # Learning
    o('Learner.maxTrainIters', 6),
  nil)
end

def webquestions
  l(
    # Data
    letDefault(:data, 0),
    sel(:data,
      l( # Webquestions (dev) [EMNLP final JSON]
        o('Dataset.inPaths',
          'train,lib/data/webquestions/dataset_11/webquestions.examples.train.json'),
        unbalancedTrainDevSplit,
      nil),
      l( # Webquestions (test) [EMNLP final JSON]
        o('Dataset.inPaths',
          'train,lib/data/webquestions/dataset_11/webquestions.examples.train.json',
          'test,lib/data/webquestions/dataset_11/webquestions.examples.test.json'),
      nil),
    nil),

    # Grammar
    letDefault(:grammar, 1),
    sel(:grammar, l(), l(o('Grammar.inPaths', 'freebase/data/emnlp2013.grammar'))),

    o('Parser.beamSize', 200), # {07/03/13}: WebQuestions is too slow to run with default 500, so set to 200 for now...

    # Caching
    letDefault(:entitysearch, 1),
    sel(:entitysearch,  # Used for EMNLP 2013
      l(
        cachePaths('lucene/0.cache', 'sparql/1.cache'),
        o('EntityLexicon.inexactMatchIndex','lib/lucene/4.4/inexact/'),
        o('LexiconFn.maxEntityEntries',10),
        o('Grammar.tags', 'webquestions', 'bridge', 'join', 'inject','inexact'), # specify also strategy
      nil),
    nil),

    # Learning
    o('Learner.maxTrainIters', 3),

    # Use binary predicate features (overfits on free917)
    o('BridgeFn.useBinaryPredicateFeatures', true),
    o('BridgeFn.filterBadDomain',true),
    letDefault(:split, 0), selo(:split, 'Dataset.splitRandom', 1,2,3),
  nil)
end


addMode('freebase', 'Freebase (for EMNLP 2013, ACL 2014, TACL 2014)', lambda { |e| l(
  letDefault(:train, 0),
  letDefault(:interact, 0),

  sel(:interact, l(), rlwrap),
  freebaseHeader,
  'edu.stanford.nlp.sempre.Main',
  freebaseOpts,

  # Dataset
  sel(:domain, {
    'webquestions' => webquestions,
    'free917' => free917,
  }),

  # Training
  sel(:train, l(), l(
    letDefault(:agenda, 0),
    sel(:agenda, l(), agendaExperiments, agendaFree917Experiments),
  nil)),

  sel(:interact, l(), l(
    # After training, run interact, which loads up a set of parameters and
    # puts you in a prompt.
    o('Dataset.inPaths'),
    o('Learner.maxTrainIters', 0),
    required(:load, 'none or exec number (e.g., 15) to load'),
    lambda { |e|
      if e[:load] == 'none' then
        l()
      else
        execPath = "lib/models/#{e[:load]}.exec"
        l(
          o('Builder.inParamsPath', execPath+'/params'),
          o('Grammar.inPaths', execPath+'/grammar'),
          o('Master.logPath', lambda{|e| 'state/' + e[:domain] + '.log'}),
          o('Master.newExamplesPath', lambda{|e| 'state/' + e[:domain] + '.examples'}),
          o('Master.onlineLearnExamples', true),
          # Make sure features are set properly!
        nil)
      end
    },
    o('Main.interactive'),
  nil))
) })

addMode('cacheserver', 'Start the general-purpose cache server that serves files with key-value maps', lambda { |e|
  l(
    'java', '-Xmx36g', '-ea', '-cp', 'libsempre/*:lib/fig.jar',
    'edu.stanford.nlp.sempre.cache.StringCacheServer',
    letDefault(:port, 4000),
    lambda { |e| o('port', e[:port]) },

    letDefault(:cachetype, 1),
    sel(:cachetype,
      l(
        o('FileStringCache.appendMode'),
        o('FileStringCache.capacity', 35 * 1024),
        o('FileStringCache.flushFrequency', 2147483647),
      nil),
      l(
        o('FileStringCache.appendMode',false),
        o('FileStringCache.capacity', 1 * 1024),
        o('FileStringCache.flushFrequency', 100000),
      nil),
    nil),
  nil)
})

############################################################
# Freebase RDF database (for building SPARQL database)

# Scratch directory
def scrOptions
  letDefault(:scr, '/u/nlp/data/semparse/rdf/scr/' + `hostname | cut -f 1 -d .`.chomp)
end

addMode('filterfreebase', '(1) Filter RDF Freebase dump (do this once) [takes about 1 hour]', lambda { |e| l(
  scrOptions,
  l(
    'fig/bin/qcreate', o('statePath', lambda{|e| e[:scr] + '/state'}),
    'java', '-ea', '-Xmx20g', '-cp', 'libsempre/*:lib/*',
    'edu.stanford.nlp.sempre.freebase.FilterFreebase',
    o('inPath', '/u/nlp/data/semparse/scr/freebase/freebase-rdf-2013-06-09-00-00.canonicalized'),
    sel(:keep, {
      'all' => o('keepAllProperties'),
      'geo' => l(
        o('keepTypesPaths', 'data/geo.types'),
        o('keepPropertiesPath', 'data/geo.properties'),
        o('keepGeneralPropertiesOnlyForSeenEntities', true),
      nil),
    }),
    o('execDir', '_OUTPATH_'), o('overwriteExecDir'),
  nil),
nil) })

addMode('sparqlserver', '(2) Start the SPARQL server [do this every time]', lambda { |e| l(
  scrOptions,
  required(:exec),
  sel(nil,
    l(
      'freebase/scripts/virtuoso', 'start',
      lambda{|e| e[:scr]+'/state/execs/'+e[:exec].to_s+'.exec/vdb'}, # DB directory
      lambda{|e| 3000+e[:exec]}, # port
    nil),
    # Give everyone permissions so that anyone can kill the server if needed.
    l(
      'chmod', '-R', 'og=u',
      lambda{|e| e[:scr]+'/state/execs/'+e[:exec].to_s+'.exec/vdb'}, # DB directory
    nil),
    # To stop the server: freebase/scripts/virtuoso stop 3093
  nil),
nil) })

# (3) Index the filtered RDF dump [takes 48 hours]
addMode('indexfreebase', '(3) Index the filtered RDF dump [takes 48 hours for Freebase]', lambda { |e| l(
  letDefault(:stage, nil),
  scrOptions,
  required(:exec),
  sel(:stage,
    l(
      'scripts/virtuoso', 'add',
      lambda{|e| e[:scr]+'/state/execs/'+e[:exec].to_s+'.exec/0.ttl'}, # ttl file
      lambda{|e| 3000+e[:exec]}, # port
      lambda{|e| e[:offset] || 0}, # offset
    nil),
    l(
      'scripts/extract-freebase-schema.rb',
      lambda{|e| 'http://localhost:'+(3000+e[:exec]).to_s+'/sparql'}, # port
      lambda{|e| e[:scr]+'/state/execs/'+e[:exec].to_s+'.exec/schema.ttl'},
    nil),
  nil),
nil) })

addMode('convertfree917', 'Convert the Free917 dataset', lambda { |e| l(
  'java', '-ea', '-Xmx15g',
  '-cp', 'libsempre/*:lib/*',
  'edu.stanford.nlp.sempre.freebase.Free917Converter',
  o('inDir','/u/nlp/data/semparse/yates/final-dataset-acl-2013-all/'),
  o('outDir','data/free917_convert/'),
  o('entityInfoFile','/user/joberant/scr/fb_data/3/entityInfo.txt'),
  o('cvtFile','lib/fb_data/2/Cvts.txt'),
  o('midToIdFile','/u/nlp/data/semparse/scr/freebase/freebase-rdf-2013-06-09-00-00.canonical-id-map'),
nil) })

addMode('query', 'Query a single logical form or SPARQL', lambda { |e| l(
  codalab,
  'java', '-ea',
  '-cp', 'libsempre/*:lib/*',
  'edu.stanford.nlp.sempre.freebase.SparqlExecutor',
  sparqlOpts,
nil) })

############################################################


# Just start a simple interactive shell to try out SEMPRE commands
addMode('simple', 'Simple shell', lambda { |e| l(
  codalab, rlwrap, 'java', '-cp', 'libsempre/*:lib/*', '-ea', 'edu.stanford.nlp.sempre.Main',
  o('Main.interactive'),
nil) })

addMode('simple-sparql', 'Simple shell for querying SPARQL', lambda { |e| l(
  codalab, rlwrap, 'java', '-Dmodules=core,freebase', '-cp', 'libsempre/*:lib/*', '-ea', 'edu.stanford.nlp.sempre.Main',
  o('executor', 'freebase.SparqlExecutor'),
  sparqlOpts,
  o('Main.interactive'),
nil) })

addMode('simple-freebase', 'Simple shell for using Freebase', lambda { |e| l(
  rlwrap, 'java', '-Dmodules=core,freebase', '-cp', 'libsempre/*:lib/*', '-ea', 'edu.stanford.nlp.sempre.Main',
  o('executor', 'freebase.SparqlExecutor'),
  letDefault(:sparqlserver, 'freebase.cloudapp.net:3093'),
  letDefault(:cacheserver, 'freebase.cloudapp.net:4000'),
  sparqlOpts,
  # Set up Freebase search for entities
  # Assume run following on the server (read-only and capacity are important!)
  #   ./run @mode=cacheserver -readOnly -capacity MAX -basePath lib/fb_data
  o('FreebaseSearch.cachePath', 'FreebaseSearch.cache'),
  o('EntityLexicon.mid2idPath', lambda { |e| e[:cacheserver] + ':freebase-rdf-2013-06-09-00-00.canonical-id-map.gz' }),
  o('TypeInference.typeLookup', 'freebase.FreebaseTypeLookup'),
  o('FreebaseTypeLookup.entityTypesPath', lambda { |e| e[:cacheserver] + ':freebase-rdf-2013-06-09-00-00.canonicalized.en-types.gz' }),
  o('EntityLexicon.maxEntries', 2),
  o('FeatureExtractor.featureDomains', 'rule'),
  o('Parser.coarsePrune'),
  o('JoinFn.typeInference'),
  o('UnaryLexicon.unaryLexiconFilePath', '/dev/null'),
  o('BinaryLexicon.binaryLexiconFilesPath', '/dev/null'),
  #o('JoinFn.showTypeCheckFailures'),  # Use this to debug
  o('Grammar.inPaths', 'freebase/data/demo1.grammar'),  # Override with your own custom grammar
  o('SparqlExecutor.returnTable'),
  #o('SparqlExecutor.includeSupportingInfo'),  # Show full information
  o('Main.interactive'),
nil) })


############################################################
# {5/27/15} [Ice]
addMode('tables', 'QA on HTML tables', lambda { |e| l(
  # Add @cldir=1 to use CodaLab's directory paths
  letDefault(:cldir, 0),
  # Usual header
  rlwrap, header('core,tables,corenlp'),
  # Select class
  letDefault(:class, 'main'),
  sel(:class, {
    'main' => 'edu.stanford.nlp.sempre.Main',
    'check' => 'edu.stanford.nlp.sempre.tables.test.DPParserChecker',
    'align' => 'edu.stanford.nlp.sempre.tables.alignment.IBMAligner',
    'dump' => 'edu.stanford.nlp.sempre.tables.serialize.SerializedDumper',
    'load' => 'edu.stanford.nlp.sempre.tables.serialize.SerializedLoader',
    'stats' => 'edu.stanford.nlp.sempre.tables.test.TableStatsComputer',
  }),
  # Fig parameters
  selo(:cldir, 'execDir', '_OUTPATH_', '.'),
  o('overwriteExecDir'), o('addToView', 0), o('jarFiles', 'libsempre/*'),
  sel(:cldir, l(), '>/dev/null'),
  # Set environment for table execution
  o('executor', 'tables.lambdadcs.LambdaDCSExecutor'),
  o('Builder.valueEvaluator', 'tables.TableValueEvaluator'),
  o('TargetValuePreprocessor.targetValuePreprocessor', 'tables.TableValuePreprocessor'),
  o('NumberFn.unitless'), o('NumberFn.alsoTestByConversion'),
  o('TypeInference.typeLookup', 'tables.TableTypeLookup'),
  o('JoinFn.specializedTypeCheck', false),
  o('JoinFn.typeInference', true),
  # Parser
  letDefault(:parser, 'floatsize'),
  sel(:parser, {
    'floatsize' => l(
      o('Builder.parser', 'FloatingParser'),
      o('useSizeInsteadOfDepth'),
      o('FloatingParser.maxDepth', 15),
    nil),
    'baseline' => l(
      o('Builder.parser', 'tables.baseline.TableBaselineParser'),
    nil),
    'dummy' => o('Builder.parser', 'tables.serialize.DummyParser'),
  }),
  o('FloatingParser.useAnchorsOnce', true),
  letDefault(:pruning, 1),
  sel(:pruning,
    l(),
    l(
      o('DerivationPruner.pruningStrategies', *tablesPruningStrategies),
      o('DerivationPruner.pruningComputers', 'tables.TableDerivationPruningComputer'),
    nil),
  nil),
  # Grammar
  tablesGrammarPaths,
  # Dataset
  letDefault(:data, 'none'),
  letDefault(:unseen, 0),
  tablesDataPaths,
  # Verbosity
  o('FeatureVector.ignoreZeroWeight'),
  o('maxPrintedPredictions', 10), o('maxPrintedTrue', 10), o('logFeaturesLimit', 10),
  letDefault(:verbose, 0),
  sel(:verbose,
    l(),
    l(
      o('showRules'),
      o('Parser.verbose', 2),
      o('JoinFn.verbose', 3),
      o('JoinFn.showTypeCheckFailures'),
    nil),
  nil),
  # Language Analyzer
  letDefault(:lang, 'corenlp'),
  sel(:lang, {
    'simple' => o('LanguageAnalyzer', 'SimpleAnalyzer'),
    'corenlp' => l(o('LanguageAnalyzer', 'corenlp.CoreNLPAnalyzer'), o('annotators', *'tokenize ssplit pos lemma ner'.split)),
    'fullcorenlp' => l(o('LanguageAnalyzer', 'corenlp.CoreNLPAnalyzer'), o('annotators', *'tokenize ssplit pos lemma ner parse'.split)),
  }),
  # Training
  letDefault(:train, 0),
  sel(:train,
    l(),
    l(
      o('combineFromFloatingParser'),
      sel(:unseen, unbalancedTrainDevSplit, l()),
      o('maxTrainIters', 3),
      o('showValues', false), o('showFirstValue'),
    nil),
    l(
      # for dumping derivations (@class=dump)
      # force unbalancedTrainDevSplit + combine from floating parser
      o('combineFromFloatingParser'), o('DPParser.cheat'),
      sel(:unseen, unbalancedTrainDevSplit, l()),
    nil),
  nil),
  # Regularization
  letDefault(:l1, 1),
  sel(:l1,
    l(),
    l(o('Params.l1Reg','lazy'), o('Params.l1RegCoeff', '3e-5')),
    l(o('Params.l1Reg','lazy'), selo(nil, 'Params.l1RegCoeff', 0, 0.00001, 0.0001, 0.001, 0.01)),
    l(o('Params.l1Reg','lazy'), selo(nil, 'Params.l1RegCoeff', 0.00001, 0.00003, 0.0001, 0.0003)),
    l(o('Params.l1Reg','lazy'), selo(nil, 'Params.l1RegCoeff', 0.00001, 0.00003, 0.0005)),
  nil),
  # Features
  letDefault(:feat, 'none'),
  sel(:feat, {
    'none' => l(),   # No features (random)
    'some' => l(     # Add your own features! (only set up the feature computers)
      o('FeatureExtractor.featureComputers', 'tables.features.PhrasePredicateFeatureComputer tables.features.PhraseDenotationFeatureComputer'.split),
    nil),
    'all' => l(      # All features
      o('FeatureExtractor.featureDomains', 'custom-denotation phrase-predicate phrase-denotation headword-denotation missing-predicate'.split),
      o('FeatureExtractor.featureComputers', 'tables.features.PhrasePredicateFeatureComputer tables.features.PhraseDenotationFeatureComputer'.split),
    nil),
    'baseline' => l( # For the baseline classifier
      o('FeatureExtractor.featureDomains', 'custom-denotation phrase-denotation headword-denotation table-baseline'.split),
      o('FeatureExtractor.featureComputers', 'tables.baseline.TableBaselineFeatureComputer tables.features.PhraseDenotationFeatureComputer'.split),
    nil),
    'ablate' => l(
      o('FeatureExtractor.featureComputers', 'tables.features.PhrasePredicateFeatureComputer tables.features.PhraseDenotationFeatureComputer'.split),
      selo(nil,
        'FeatureExtractor.featureDomains',
        'phrase-predicate phrase-denotation headword-denotation missing-predicate'.split,
        'custom-denotation phrase-denotation headword-denotation missing-predicate'.split,
        'custom-denotation phrase-predicate headword-denotation missing-predicate'.split,
        'custom-denotation phrase-predicate phrase-denotation missing-predicate'.split,
        'custom-denotation phrase-predicate phrase-denotation headword-denotation'.split,
      nil),
    nil),
  }),
  letDefault(:featOp, 'careful'),
  sel(:featOp, {
    'none' => l(),
    'base' => l(
      o('usePredicateLemma'), o('usePhraseLemmaOnly'),
    nil),
    'careful' => l(
      o('usePredicateLemma'), o('usePhraseLemmaOnly'),
      o('maxNforLexicalizeAllPairs', 2),
      o('traverseWithFormulaTypes'), o('reverseNameValueConversion', 'allBang'),
      o('lookUnderCellProperty'), o('useGenericCellType'),
      o('computeFuzzyMatchPredicates'),
    nil),
  }),
nil) })

def tablesGrammarPaths
  lambda { |e|
    baseDir = ['tables/grammars/', 'grammars/'][e[:cldir]]
    l(
      letDefault(:grammar, 'combined-all'),
      sel(:grammar, {
        'restrict' => o('Grammar.inPaths', "#{baseDir}restrict.grammar"),
        'simple' => o('Grammar.inPaths', "#{baseDir}simple.grammar"),
        'combined' => o('Grammar.inPaths', "#{baseDir}combined.grammar"),
        'combined-jnc' => l(    # WQ baseline
          o('Grammar.inPaths', "#{baseDir}combined.grammar"),
          o('Grammar.tags', *'movement count'.split),
        nil),
        'combined-cut' => l(    # No intersection / union
          o('Grammar.inPaths', "#{baseDir}combined.grammar"),
          o('Grammar.tags', *'movement comparison count aggregate superlative arithmetic'.split),
        nil),
        'combined-all' => l(    # Default
          o('Grammar.inPaths', "#{baseDir}combined.grammar"),
          o('Grammar.tags', *'alternative movement comparison count aggregate superlative arithmetic merge'.split),
        nil),
        'combined-trigger' => l(    # Use trigger words for operations
          o('Grammar.inPaths', "#{baseDir}combined.grammar"),
          o('Grammar.tags', *'t-alternative t-movement t-comparison t-count t-aggregate t-superlative t-arithmetic merge'.split),
        nil),
      }),
    )
  }
end

def tablesDataPaths
  lambda { |e|
    baseDir = ['lib/data/tables/data/', 'SQA/data/'][e[:cldir]]
    csvDir = ['lib/data/tables/', 'SQA/'][e[:cldir]]
    datasets = {
      'none' => l(),
      # Pristine test test
      'test' => l(
        o('Dataset.inPaths',
          "train,#{baseDir}train.examples",
          "test,#{baseDir}test.examples"),
          #"test,#{baseDir}pristine-unseen-tables.examples"),
        o('Dataset.trainFrac', 1), o('Dataset.devFrac', 0),
        let(:unseen, 1),
      nil),
    }
    # Development sets: 80:20 random split of training data
    ['1', '2', '3', '4', '5'].each do |x|
      datasets['u-' + x] = l(
        o('Dataset.inPaths',
          "train,#{baseDir}random-split-#{x}-train.examples",
          "dev,#{baseDir}random-split-#{x}-dev.examples",
        nil),
        let(:unseen, 1),
      nil)
    end
    # That's it!
    l(
      o('TableKnowledgeGraph.baseCSVDir', csvDir),
      sel(:data, datasets),
    nil)
  }
end

def tablesPruningStrategies
  [
    # Formula
    "singleton",
    "multipleSuperlatives",
    "sameMerge",
    "forwardBackward",
    "doubleNext",
    # Denotation
    "emptyDenotation",
    "nonLambdaError",
    #"tooManyValues",
    "badSuperlativeHead",
    "mistypedMerge",
  nil].compact
end


############################################################

if ARGV.size == 0
  puts "#{$0} @mode=<mode> [options]"
  puts
  puts 'This is the main entry point for all SEMPRE-related runs.'
  puts "Modes:"
  $modes.each { |name,description,func|
    puts "  #{name}: #{description}"
  }
end

modesMap = {}
$modes.each { |name,description,func|
  modesMap[name] = func
}
run!(sel(:mode, modesMap))
