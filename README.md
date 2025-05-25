# Concept Drift Robust Policy Learning

This repo provides the replication code of the real-world dataset experiment in the paper "Distributionally Robust Policy Learning under Concept Drifts".

We consider the dataset of a large-scale randomized experiment comparing assistance programs offered to French unemployed individuals provided in Behaghel et al., 2014. The decision maker is trying to learn a personalized policy that decides whether to provide: (i) an intensive counseling program run by a public agency; or (ii) a similar program run by private agencies, to an unemployed individual. The processed dataset is first provided by Kallus, 2023.

Note that in this example, since the dataset is collected from a randomized experiment, the propensity score is assumed to be constant (but unkown to the learning algorithm LN) across the population.

---

## Environment Setup

Create an environment and install requirements.

```
conda create -n conceptdriftLN python=3.10
conda activate conceptdriftLN
pip install -r requirements.txt
```

## Dataset Generation

The training and testing datasets are generated using the following code for testing LN over x dataseeds.

```
python generate_data.py x
```

In our experiment, we set x=50 to run the experiment over 50 seeds.

## Policy Learning Test

To learn our concept drift distributionally robust policy LN using the training and testing dataset generated with seed x,
run the following command.

```
python learn_ln.py x
```

The default algorithm parameters, which we used to generate our results, are also outlined in learn_ln.py.

In our experiments, all 50 seeds of datasets are tested, and the average performance (including 95% confidence intervals) of LN are reported.

## Reference

Behaghel, L., Crépon, B., and Gurgand, M.
<b>Private and public provision of counseling to job seekers: Evidence from a large controlled experiment.</b>
<i>American economic journal: applied economics,</i> 6(4):142-174, 2014.

Kallus, N.
<b>Treatment effect risk: Bounds and inference.</b>
<i>Management Science,</i> 69(8):4579-4590, 2023.

Sverdrup, E., Kanodia, A., Zhou, Z., Athey, S., and Wager, S.
<b>policytree: Policy learning via doubly robust empirical welfare maximization over trees.</b>
<i>Journal of Open Source Software,</i> 5(50):2232, 2020.
