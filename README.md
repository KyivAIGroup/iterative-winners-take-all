# Iterative winners-take-all

This repository contains Python implementation of our paper

**FORMATION  OF CELL ASSEMBLIES WITH ITERATIVEWINNERS-TAKE-ALL COMPUTATION ANDEXCITATIONâ€“INHIBITION  BALANCE**

available on arxiv at https://arxiv.org/abs/2108.00706.

## Quick start

Python 3.6+ is required.

First, install the [requirements](./requirements.txt):
```
pip install -r requirements.txt
```

Then you're ready to run the scripts to reproduce the figures in the paper.

For PyTorch implementation, refer to the [`nn`](./nn) directory.

Should you have any questions, please [open an issue](https://github.com/KyivAIGroup/iterative-winners-take-all/issues) or email us.

## Reproducing the figures

To reproduce the plots, run `python figX.py` in a terminal, where `figX.py` is one of the following:

* [`fig2.py`](./fig2.py) - how the output populations sparsity depends on the weight sparsity (weights are random fand fixed);
* [`fig3b.py`](./fig3b.py) - habituation;
* [`fig4.py`](./fig4.py) - clustering;
* [`fig5.py`](./fig5.py) - dependence on the input density (not shown in the paper);
* [`fig6.py`](./fig6.py) - similarity preservation (not show in the paper);
* [`decorrelation.py`](./decorrelation.py) - decorrelation (not shown in the paper).

This section compiles all figures, listed and supplementary, in one page.
### Figure 2 (from the paper). Dependence of the density of encodings on the weight density
The weights are random and fixed.
<table style="width:100%">
    <tr>
        <td>
            <img src="figures/fig2a.png"/>
        </td>
        <td>
            <img src="figures/fig2b.png"/>
        </td>
    </tr>
</table>

The figures from the next sections involve learning. *Learning* the weights means letting the network evolve over time from input stimuli `x` rather than "training" it from labelled data. In other words, the learning is local and unsupervised.

### Habituation

<table style="width:100%">
    <tr>
        <td>
            <img src="figures/habituation/simpleHebb.png"/>
        </td>
        <td>
            <img src="figures/habituation/permanence-fixed.png"/>
        </td>
        <td>
            <img src="figures/habituation/permanence-varying.png"/>
        </td>
		<td>
            <img src="figures/habituation/permanence-vogels.png"/>
        </td>
    </tr>
</table>

### Decorrelation

<table style="width:100%">
    <tr>
        <td>
            <img src="figures/decorrelation/simpleHebb.png"/>
        </td>
        <td>
            <img src="figures/decorrelation/permanence-fixed.png"/>
        </td>
        <td>
            <img src="figures/decorrelation/permanence-varying.png"/>
        </td>
		<td>
            <img src="figures/decorrelation/permanenceVogels.png"/>
        </td>
		<td>
            <img src="figures/decorrelation/kWTA-permanence-fixed.png"/>
        </td>
    </tr>
</table>


### Clustering

Comparison of iWTA (with *permanence-varying* learning) and kWTA on a clustering task. Run the [fig4.py](./fig4.py) script to generate this plot.

<img src="figures/clustering/comparison.png" height="500"/>

Complete run of each learning method:

<table style="width:100%">
    <tr>
        <td>
            <img src="figures/clustering/simpleHebb.png"/>
        </td>
        <td>
            <img src="figures/clustering/permanence-fixed.png"/>
        </td>
        <td>
            <img src="figures/clustering/permanence-varying.png"/>
        </td>
		<td>
            <img src="figures/clustering/permanenceVogels.png"/>
        </td>
		<td>
            <img src="figures/clustering/kWTA-permanence-fixed.png"/>
        </td>
    </tr>
</table>


### Figure 5. Dependence on the input density

<img src="figures/fig2c.png" alt="figures/fig2c.png" width="400"/>

Dependence of the density of encodings on the input density.
The main feature is that the dependence is nonlinear and the same for different layer sizes.

### Figure 6. Similarity preservation

<img src="figures/similarity_preservation.png" alt="figures/similarity_preservation.png" width="400"/>

The figure shows how the cosine similarity between two input vectors changes in the encoding space. The sparsity of encodings influences the dependence. Results for iWTA are comparable to kWTA. Similar inputs are encoded in similar output populations. The higher the encoding density (lower sparsity) the better preservation.
