import numpy as np
import pandas as pd
import scipy.special
import scipy.stats as st
import os

import cmdstanpy
import arviz as az

import iqplot

import colorcet
import bokeh.io
import bokeh.plotting
bokeh.io.output_notebook()
df = pd.read_csv(os.path.join('E:/ALI/processing project/Bayesian_inference', 'singer_transcript_counts.csv'), comment='#')
genes = ["Nanog", "Prdm14", "Rest", "Rex1"]

plots = [
    iqplot.ecdf(
        data=df[gene].values,
        q=gene,
        x_axis_label="mRNA count",
        title=gene,
        frame_height=150,
        frame_width=200,
    )
    for gene in genes
]
bokeh.io.show(bokeh.layouts.gridplot(plots, ncols=2))

stan_code="""data {
  int<lower=0> N;
  int<lower=0> n[N];
}


parameters {
  real log10_alpha;
  real log10_b;
}


transformed parameters {
  real alpha = 10^log10_alpha;
  real b = 10^log10_b;
  real beta_ = 1.0 / b;
}


model {
  // Priors
  log10_alpha ~ normal(0, 1);
  log10_b ~ normal(2, 1);

  // Likelihood
  n ~ neg_binomial(alpha, beta_);
}"""
with open("gene.stan", "w") as f:
    f.write(stan_code)
    
sm = cmdstanpy.CmdStanModel(stan_file='gene.stan')
data = dict(N=len(df), n=df["Rest"].values.astype(int))
samples = sm.sample(
    data=data,
    chains=4,
    iter_sampling=1000,
)
samples = az.from_cmdstanpy(posterior=samples)
