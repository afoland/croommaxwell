# croommaxwell
An LSTM system for training on and generating poetry

Derived from a repository by Larspars , in turn based on a set of Oxford exercises https://github.com/oxford-cs-ml-2015/practical6 . Neither has, at that time or as of the first commit in this repository, license terms.

Work done mostly 2016-2018, log files included.

Primary modifications were in scripts to create curriculum training, and use of "nearest word" (by levenshtein distance) for initializing embedding unrecognized words.

The system operated on a mix of Gutenberg text to start (primarily Victorian novelists), graduating to all-poetry.  Poetry was romantics (Tennyson, Keats, Dickenson, etc the usual suspects)

Approximately 155M parameters in final form.

Two anthologies of poems are included.

# About C. Room Maxwell
Room is found in the northern climes of Wellesley, Massachusetts, spending time reading,
writing, and hibernating. A poetic autodidact, Room finds inspiration primarily in the romantic poets
and Victorian novelists, but takes insight from poetry and prose whereever it can be found to inform an idiosyncratic poetic style. Other works include a collection of short stories, and a novel currently in search of a publisher.
