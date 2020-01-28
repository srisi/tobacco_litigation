# Focus on the Facts to Hide the Truth: A Quantitative Analysis of the Tobacco Industry Rhetoric in Cigarette Litigation

This repository contains the code used to generate our (Stephan Risi and Robert Proctor's)
analysis of the tobacco industry's courtroom rhetoric. 
In total, our corpus consists of 318 closings from 159 Engle Progeny cases, consisting of more
than 10,000,000 words. 

Our findings are published in Tobacco Control:[Stephan Risi and Robert Proctor. "Big Tobacco Focuses on the Facts to Hide the Truth: an Algorithmic Exploration of Courtroom Tropes and Taboos," Tobacco Control. Published Online First: 13 September 2019. doi: 10.1136/tobaccocontrol-2019-054953.](https://tobaccocontrol.bmj.com/content/early/2019/09/27/tobaccocontrol-2019-054953)

You can find an interactive visualization of the results on 
[Tobacco Analytics.](http://www.tobacco-analytics.org/litigation)

As of March 2019, you can search for both individual terms and [Linguistic Inquiry and Word Count
(LIWC) categories](http://liwc.wpengine.com/). 


## Code Organization
This repository contains all of the code as well as the results from our analysis of tobacco 
courtroom rhetoric. Hence, 

If you are interested in re-running the experiment, you can use `create_json_dataset()` in 
`dataset_creation.py`, which calculates all of the results, stores them in the sqlite database
`distinctive_tokens.db` and creates the `distinctive_tokens.json`, which we use to display 
results on [Tobacco Analytics.](http://www.tobacco-analytics.org/litigation)

`corpus.py` and `closing.py` instantiate classes for the whole corpus and individual closings
respectively. `stats.py` contains the code used to calculate frequency ratios, Dunning's 
Log-Likelihood scores, and Mann-Whitney Rho scores.
 

## Data and Data Organization



#### Metadata

You can find the metadata for all 159 represented trials in data/litigation_dataset_master.csv. 
For each trial, it contains the following fields:
- Case    (The case name)
- Case ID (The case ID that we have assigned, based on the last name of the plaintiff, e.g. 
 "ahrens1")
 - Trial Date
 - Phase (The trial phase. Note: This study only includes phase 1 trials. Phase 0 indicates that 
 we didn't have data on a potential phase 2)
 - Type (Always "closing." We might study openings in a second study)
 - Side ("plaintiff" or "defendant")
 - Filename (The automatically generated filename for each closing. It consists of 
 `<case_id>_<phase>_<opening/closing>_<side>.txt`. e.g. `ahrens1_1_c_d.txt` means it's the 
  first case involving a plaintiff named Ahrens, phase 1 of the trail, the "c" indicates a 
  closing statement and "d" means it's the defense's closing statement)
 - Doc date (The date of the closing statement)
- TID: The document ID assigned by the 
[Truth Tobacco Documents Library](https://www.industrydocumentslibrary.ucsf.edu/tobacco/)

Beyond these fields, you can also find further information on the outcome of the trial, the 
represented companies, and their counsels.

#### Full Text
You can find the full text for each closing statement in the data/closings folder by using the
filename from the metadata.

Each closing statement appears in three forms:

- text_raw contains the original text without cleaning
- text_clean is a clean version of the closing with expanded contractions and all in lower case
- part_of_speech contains the part-of-speech tags of the document.
