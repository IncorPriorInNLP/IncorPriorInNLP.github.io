# Knowledge Base

1. TOC
{:toc}

## Injecting Logical Background Knowledge into Embeddings for Relation Extraction (NAACL 2015)

### Rule Formula
$r_s(x,y) \Rightarrow r_t(x, y)$

$(x,y)$ is entity pair and $r_s, r_t$ are relations

### Rule Source
We first run matrix factorization
over the complete training data to learn
accurate relation and entity-pair embeddings. After training, we iterate over all pairs of relations $(r_s, r_t)$ where $r_t$ is a Freebase relation. For every relation pair
we iterate over all training atoms $r_s(e_i, e_j)$, evaluate the score $[r_s(e_i, e_j) \Rightarrow r_t(e_i, e_j )]$ using t-norm, and calculate the average to arrive at a score for the formula. Finally, we rank all formulae
by their score and manually filter the top 100 formulae, which resulted in 36 annotated high-quality formulae.

### How to incorporate rules
Modify the objective function. Still use pairwise-ranking function:

$L = L(S(f^+) - S(f^-))$

$S$ is a score function, $f$ means $r(e_i, e_j)$ in original loss. In this work, $f$ can also means $r_s(x,y) \Rightarrow r_t(x, y)$. 

$S$ for $r(e_i, e_j)$ follows the basic model, and $S$ for $r_s(x,y) \Rightarrow r_t(x, y)$ is based on ***t-norm***:

$S(f_1 \Rightarrow f_2) = S(f_1)S(f_2)-S(f_1)+1$

### Why Work?
* Provide data in zero-shot setting.
* More than providing more data in few-shot. As simply adding inferred atom via first-order logic into train data does not work.
* Prevent model to capture $r_t \Rightarrow r_s$ if $r_s \Rightarrow r_t$ holds. *It is really confusing ...*