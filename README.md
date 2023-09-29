# croommaxwell
An LSTM system for training on and generating poetry

Derived from a repository by Larspars , in turn based on a set of Oxford exercises https://github.com/oxford-cs-ml-2015/practical6 . Neither has, at that time or as of the first commit in this repository, license terms.

Work done mostly 2016-2018, log files included.

Primary modifications were in scripts to create curriculum training, and use of "nearest word" (by levenshtein distance) for initializing embedding unrecognized words.

The system operated on a mix of Gutenberg text to start, graduating to all-poetry.  Poetry was romantics (Tennyson, Keats, Dickenson, etc the usual suspects)

Approximately 155M parameters in final form.

Two anthologies of poems are included.
