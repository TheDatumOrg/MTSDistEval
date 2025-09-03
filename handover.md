# SIGMOD reproducibility submission

## What they ask us to provide
- Include detailed readme and scripts that can (i) run experiments, (ii) collect and parse data, and (iii) visualize  the results by plotting your paper graphs. If some data  collection needs to be manual  (ideally, it should not), please  add detailed step-by-step instructions on how to do so.
- Before submitting your package, please try to recreate all your guidelines  to a fresh machine–that way, you will probably face  any  problems the reviewers might face, and you can minimize them  prior to  submission.
- If the submitted scripts do not reproduce some figures, add a  concrete  explanation as to why the core thesis of the paper can  be  verified without  the missing experiments.
- Include a detailed description of the hardware used.

## What I did today
1. Added a script called `scripts/run_classification_exp.sh` that runs all classification experiments with different distance measures and normalization techniques needed to generate the plots and tables in our paper. 
2. Added another script called `scripts/generate_plots.py` that generates the plots from the results of the classification experiments. This script was based on the `analysis/evaluation_nonorm.ipynb` notebook in our original repo. Note that tables are written to a text file called plots/tables.txt
3. Wrote a description of the hardware used for our experiments (i.e., the specs of the dutch supercomputing cluster). I added this in the current README.md

> Note that, to test if the run_classification_exp.sh script works correctly, I added a `--testrun` flag to the arguments of `classification.py` that, when set, runs classification on a small dummy dataset with only a few vectors and randomly generates the distance matrix rather than actually compute it. This makes sure that every run only takes a few ms, which enables me to verify if all commands worked and also generate the necessary output files in order to also test the plotting script. In other words, within a few minutes, you would get a csv with all results for all normalizations, datasets, measures, etc. with effectively random accuracies. To do the same for clustering and anomaly detection experiments, similar modifications will be needed.

## What we still need (i.e., TODO list)
1. Implement the `--testrun` flag for the clustering and anomaly detection experiments in `scripts/run_classification_exp.sh` to enable quick testing.
2. Add the necessary commands for running the clustering and anomaly detection experiments to `scripts/run_classification_exp.sh` (and probably rename it to something like `scripts/reproduce_paper.sh`)
3. Add the necessary logic to `scripts/generate_plots.py` to generate the plots/tables for the clustering and anomaly detection experiments.
4. Write a small standalone README file only for the SIGMOD testers that directly points them to (i) the scripts to run and visualize the experiments, and (ii) how to download and preprocess the data. This README should also include the hardware specifications I wrote in the main README.md.
5. Submit the artifacts. See the instructions email I just forwarded to Fan and Haojun.