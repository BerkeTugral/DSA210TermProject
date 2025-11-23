# DSA210 Term Project

Mustafa Berke Tuğral - 30295
28 Oct 2025

## Motivation

This project idea emerged during a discussion about Cyberpunk 2077 by CD Projekt Red. a game that, despite arguably being one of the best ever made, suffered a disastrous launch due to bugs and incomplete features. Its initial reception was so poor that many players avoided it for years, only returning once the developers had significantly improved it upon the following years and ran extensive PR campaigns about the newly acquired quality of the game. This made me think about how much player ratings influence purchasing decisions: no matter how intriguing a game appears to be, the collective judgment of other players more often than not shapes our perception of its quality and whether we are willing to spend money on purchasing it. However, a counter argument also arose in my mind: I once left a negative Steam review for Mount & Blade II: Bannerlord out of frustration, yet kept playing the game for hundreds of hours without ever updating my review and even bought it as gifts for other people. This seemingly irrational behavior of mine led me to the question: how reliable are reviews, particularly metacritic scores, in predicting the commerical success of a videogame.?

## Data Source

I collected the data from Kaggle, due to the sites accessible and organized nature. The "Video Game Sales" Database has data for more than 16,500 which is more than plenty for my project. The metacritic data also has a sufficient amount of data to compare. Also, both datasets have many common fields which will make the joining and cleaning much simpler.

Video Game Sales:
https://www.kaggle.com/datasets/gregorut/videogamesales/data

Metacritic Video Games Data:
https://www.kaggle.com/datasets/brunovr/metacritic-videogames-data?resource=download

The relevant features include the game title, release year, platform, publisher, genre, and global sales in the Video Game Sales dataset. The Metacritic dataset includes critic scores, user scores, release year equivalents such as the “r-date” field, developer names, and platforms. These shared fields will form the basis for the merge.

## Data Preparation

As of this point I do not possess the capabilities to give a very detailed explanation on how the analysis will be made, but very crudely it can be explaind like:

Step I: Acquiring the Data

I will download the raw data from kaggle

Step II: Merging and Cleaning

I will identify some keys to join them. Since both tables share many common merger points such as the name, year etc. It will be quite simple
However, it is easy to observe several naming mismatches on common features such as Video Game Sales having Year and Metacritic Video Games Data having r-date to denote the same thing.
I will properly align the data with each other before merging to avoid problems from aforementioned mismatches.

After merging, I will clean the data by removing missing or inconsistent entries, correcting duplicated names, and resolving cases where different editions of the same game appear under slightly different titles.

# Methodology

The general plan is to examine whether critic and user scores correlate with commercial success. I will translate this into measurable variables taken directly from the datasets—for example, critic score, user score, global sales, release year, platform, and publisher reputation. After cleaning and merging, I will explore the relationships between these variables using descriptive statistics and then attempt to model whether the scores can predict sales. This may involve correlation analysis, simple regression experiments, and visual inspection of patterns across different game genres or publishers. Since this is an exploratory project, the analysis does not have to be extremely complex—as long as it can answer the basic question of whether scores meaningfully influence commercial outcomes.

# Expected Outcomes

As an aspiring video game developer, I believe quantifying consumer behavior and acting towards the findings is equally as important as producing a well designed game. The curation of a good review catalogue might make an average game more commercially successful than a well developed game with bad reception. This project is expected to study the consumer behavior specifically how does to consumers react to the ratings of a game. 

My main expections are:

I:   Bad ratings are a significant factor in sales loss for videogames. 

II:  Bad ratings can turn a group of reluctant customers into an angry group who will refuse to purchase the game.

III: Lower rated games of previously acclaimed and successful studios will sell less than their higher rating games.


