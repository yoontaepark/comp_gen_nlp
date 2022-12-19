# comp_gen_nlp: Project for DSGA-1011 (NLP)

### Authors:
Yoon Tae Park <yp2201@nyu.edu> <br>
Yoobin Cheong <yc5206@nyu.edu> <br>
Yeong Koh <yeong.koh@nyu.edu> <br>
(Mentored by Najoung Kim <najoung.kim@nyu.edu>) <br>
<br>


Description:
- Final project for New York University's NLP project (DS-GA 1011, Fall 2022) mentored by Najoung Kim
- Due: Dec 14th, 2022

<!-- Criteria [here](///) -->

#### Repository Structure
```
Project
├── LICENSE
├── README.md         
├── data: 9 different datasets from multiple sources 
│   ├── SCAN(simple, length, add-turn-left), CFQ(mcd1), smcalflow: source from huggingface
│   ├── COGS(test, lexcial, structural): from github repo (https://github.com/najoungkim/COGS)
│   └── PCFG SET: from github repo (https://github.com/i-machine-think/am-i-compositional)
│
├── notebooks: store your Jupyter Notebooks here.
│   ├── yp2201: treat this workspace as your own.
│   │           feel free to add subfolders or files as you see fit.
│   │           work in a way that enables YOU to succeed
│   ├── yc5206 
│   ├── yk2678 
│   └── shared: place anything you want to share with the group, or the final version of your notebook here.
│
├── src: store your source code (e.g. py, sh) here. 
│
├── reports: Generated analysis as HTML, PDF, LaTeX, etc
│   ├── interim: any intermediate report that has been created
│   ├── figure: graphics and figures needed for report
│   └── final: final report
│
├── references: data dictionaries, manuals, and all other explanatory materials.         
├── requirements.txt: the requirements file for reproducing the analysis environment(TBD)
└── setup.py: make this project pip installable with `pip install -e` (TBD)

```
This file structure is derived from [Cookiecutter Project Template](https://drivendata.github.io/cookiecutter-data-science/).
