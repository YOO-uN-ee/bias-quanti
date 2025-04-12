Paper in work:
https://www.overleaf.com/project/67f80a42dd01f0be58a20d16

Models to test on:

DistilBERT
distilbert/distilbert-base-uncased

BERT base
google-bert/bert-base-uncased

BERT large
google-bert/bert-large-uncased

RoBERTa base
FacebookAI/roberta-base

RoBERTa large
FacebookAI/roberta-large

AlBERT base
albert/albert-base-v2

AlBERT large
albert/albert-large-v2

DeBERTa

ModernBERT base
answerdotai/ModernBERT-base

ModernBERT large
answerdotai/ModernBERT-large

Demographics:
For the demogrpahics we follow the minimum race/ethnicity reporting categories of the US Census Bureau
(Table 1 on the website: https://www.census.gov/newsroom/blogs/random-samplings/2024/04/updates-race-ethnicity-standards.html)
According to the US Census Bureau webpage: The Census Bureau collects racial data in accordance with guidelines provided by the U.S. Office of Management and Budget (OMB)

American Indian/Alaska Native
Asian
Black/African American
Hispanic/Latino
Middle Eastern/North Africa
Native Hawaiian/Pacific Islander
White

(More than Minorities and Majorities paper put it on 6 targets)
- Black
- White
- Latino
- Asian
- Native Hawaiian (yet not seen)
- American Indian (yet not seen)

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html

Reference Github repos:
https://github.com/sciphie/bias-bert
https://github.com/haozhe-an/DD-GloVe/blob/main/embeddings_eval/utils.py

We will like to measure the level of bias towards each race by looking into their performance on two downstream tasks. (Performance in naive and trained version).
If the performance difference is propagated through the downstream task, it will demonstrate that there is certain leaning in towards bias for certain race in each of these models.
The line of 'race' will be based on that defined by the US Census Bureau.
We will test it on two downstream tasks: sentiment analysis and hate speech detection towards the given race.
The difference will be measured by calculating the perplexity (the likelihood it will generate certain race in the question)
We will measure the F-statistics (ANOVA) i.e., multi class t-score. Our null hypothesis is that there is no leaning bias (bias difference towards each race) for each of these models.
If the F-statistics is high it will show that there is a leaning bias.