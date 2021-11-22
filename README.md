# Arbitrary_Distribution_Modeling

This repo implements the *Neighborhood Likelihood Loss* (**NLL**) and *Arbitrary Distribution Modeling* (**ADM**, with InteractingLayer of AutoInt as the 2-order *FeatEx<sub>2</sub>* as an example) for *Arbitrary Distribution Modeling with Censorship in Real Time Bidding Advertising* for TheWebConf'22.

*nll_adm.py* is the core codes for *NLL* and *ADM*.

*ipinyou_example.ipynb* shows an executable project to perform experiments on iPinYou dataset.

*yoyi_example.ipynb* shows an executable project to perform experiments on YOYI dataset.

This is a version of implementation where the breadth of neighbourhood is equal to one time of the difference between the bidding price and the winning price. That is r<sup>r</sup><sub>win</sub> = r<sup>l</sup><sub>win</sub> = 1.
