# Knowledge Base

## Injecting Logical Background Knowledge into Embeddings for Relation Extraction (NAACL 2015)

### Rule Formula
$r_s(x,y) \Rightarrow r_t(x, y)$

$(x,y)$ is entity pair and $r_s, r_t$ are relations

### Rule Source
We first run matrix factorization
over the complete training data to learn
accurate relation and entity-pair embeddings. After training, we itera  te over all pairs of relations $(r_s, r_t)$ where $r_t$ is a Freebase relation. For every relation pair
we iterate over all training atoms $r_s(e_i, e_j)$, evaluate the score $[r_s(e_i, e_j) \Rightarrow r_t(e_i, e_j )]$ using t-norm, and calculate the average to arrive at a score for the formula. Finally, we rank all formulae
by their score and manually filter the top 100 formulae, which resulted in 36 annotated high-quality formulae.

### How to incorporate rules
Modify the objective function. Still use pairwise-ranking function:

$L = \sigma(S(f))$

$S$ is a score function, $f$ means $r(e_i, e_j)$ in original loss. In this work, $f$ can also means $r_s(x,y) \Rightarrow r_t(x, y)$. 

$S$ for $r(e_i, e_j)$ follows the basic model, and $S$ for $r_s(x,y) \Rightarrow r_t(x, y)$ is based on ***t-norm***:

$S(f_1 \Rightarrow f_2) = S(f_1)S(f_2)-S(f_1)+1$

### Why Work?
* Provide data in zero-shot setting.
* More than providing more data in few-shot. As simply adding inferred atom via first-order logic into train data does not work.
* Prevent model to capture $r_t \Rightarrow r_s$ if $r_s \Rightarrow r_t$ holds. *It is really confusing ...*




## Regularizing Relation Representations by First-order Implications (ACL 2016 Workshop)

This is an extension of *Injecting Logical Background Knowledge into Embeddings for Relation Extraction*. These two papers share same author. The motivation is previous work need a term for every $r(e_s, e_t)$, which is costly. This work tries to model the order between two relations directly.

### Rule Formula
$r_s \Rightarrow r_t$

### Rule Source
artificial dataset

### How to incorporate rules
First map representation of entity and relation into a new space **$v$**, $(0, 1)^k$.

$v(r_s)^i > v(r_t)^i$ if $r_s \Rightarrow r_t$

$L_I^{r_s \Rightarrow r_t} = \sum_{i=1}^k log(1 + ReLU(v(r_s)^i - v(r_t)^i))$ 

Add $L_I$ in original objective function.

### Why Work?
* In artificial dataset, learned relation representations do shows the order.
* Serve as regularizer, preventing model from overfitting.

## Lifted Rule Injection for Relation Embeddings (EMNLP 2016)
This is an extension of *Injecting Logical Background Knowledge into Embeddings for Relation Extraction*. Have same author as previous two papers. Motivation is the same as last one, trying to model the order between two relations directly. 

### Rule Formulation
$r_s \Rightarrow r_t$

### Rule Source
* Same as *Injecting Logical Background Knowledge into Embeddings for Relation Extraction*. This is for few-shot
* We use WordNet hypernyms
to generate rules for the NYT dataset. To
this end we iterate over all surface form patterns in the dataset and attempt to replace words in the pattern by their hypernyms. If the resulting pattern is contained in the dataset, we generate the corresponding rule. For instance, we generate a rule appos->diplomat->amod $\Rightarrow$ appos->official->amod since both patterns
are contained in the NYT dataset and we know from WordNet that a diplomat is an official. This leads to 427 rules fromWordNet that we subsequently annotate manually to obtain 36 high-quality rules.

### How to incorporate rules
Constrain entity representation into a approximate binary space:
$v (e) = \sigma(e)$

For a pair relation $r_t$ and $r_s$, the previous work needs to add one term for every entity pair. Now use one upper bound instead.

$L_I = \sum_{t \in T}l_I([r_s - r_t]^\top \tilde t)$

$L_I = \sum_{t \in T}l_I(\sum_{i=1}^k \tilde t[r_t - r_s]^\top \mathbf{1}_i)$

$L_I \leq \sum_{i=1}^k l_I(r_s - r_t)^\top \mathbf{1}_i \sum_{t \in T} \tilde t$ 

$L_I \leq \beta \sum_{i=1}^k l_I(r_s - r_t)^\top \mathbf{1}_i$

$L_I \leq \beta L_I^U$

$r_s$ and $r_t$ are relations, $t$ is entity pair representation, $\tilde t = \frac{t}{||t||}$, and $l_I$ is a convex function.

Final objective function is the combination of original pairwise ranking loss for triple and $L_I^U$

### Why Work?
* It is better than *Injecting Logical Background Knowledge into Embeddings for Relation Extraction*. This paper claim it is because they use Bayesian Personalized Ranking (just pair-wise ranking loss). However, Bayesian Personalized Ranking is only applied to triple loss instead of the rule loss. It is confusing.
* This shows that injecting such rules influences the relation embedding space beyond only the relations explicitly stated in the rules. For example, injecting the rule ppos<-father->appos $\Rightarrow$ poss<-parent->appos can contribute
to improved predictions for the test relation
parent/child.
* Relations have order
* Asymmetric Character of Implications, if $r_s \Rightarrow r_t$ holds and $r_t(e_1, e_2)$ in train, then $r_s(e_1, e_2)$ will receive low score. 


## Jointly Embedding Knowledge Graphs and Logical Rules (EMNLP 2016)
It is nearly the same as *injecting Logical Background Knowledge into Embeddings for Relation Extraction*, except the previous work consider entity pair $(x, y)$ as a whole and this work consider each entity separately. I think this is because the change of basic model for relation extraction.

### Rule Formulation
$(x, r_s, y) \Rightarrow (x, r_t, y)$

$(x, r_{s1}, y) \land (x, r_{s2}, y) \Rightarrow (x, r_t, y)$

### Rule Source
we first run TransE to get entity and relation embeddings, and calculate the truth value for each of such rules. Then we rank all such rules by their truth values and manually filter those ranked at the top. We finally creat 47 rules on FB122, and 14 on WN18 (see Table 2 for examples).

### How to incorporate rules
Modify objective fuction, still use pairwise ranking loss:
$[\gamma - I(f^+) - I(f^-)]^+$

$f$ used to be $(e_i, r_k, e_j)$ only, but in this work, it stands for both atom and rules.

$I(e_i, r_k, e_j) = 1 - \frac{1}{3\sqrt{d}}||e_i + r_k - e_j||_1$

$I(f_1 \Rightarrow f_2) = I(f_1)I(f_2)-I(f_1)+1$

$I(f_1) \land I(f_2) = I(f_1)I(f_2)$

### Why Work？
* More than data argumentation.
* Do better for triples whose relations have been seen in rules. (The exp does not convince me ... )


## Improving Knowledge Graph Embedding Using Simple Constraints

### Rule Formulation
$(x, r_p, y) \Rightarrow (x, r_q, y)$

The base model is ComplEx, this paper requires

$0 \leq Re(e), Im(e) \leq 1, \forall e \in \varepsilon$

Then 

$Re(r_p) \leq Re(r_q), Im(r_p) = Im(r_q)$

### Rule Source
AMIE+: One model automatically extracts entailment relation. For every relation pair, it has one confidence score. Use the rules whose score are higher than 0.8 .

### How to incorporate rules
Truncate embedding representation into (0, 1) after updating parameters each time.

Modify the objective function:

$\sum log(1+exp(-y_{ijk}\phi (e_i, r_k, e_j))) + \mu \sum \lambda 1^T[Re(r_p) - Re(r_q)]_{+} + \mu \sum \lambda 1^T (Im(r_p) - Im(r_q))^2$

### Why Work?
* Cluster for entity representation.
* Case study shows $Re(r_p) \leq Re(r_q), Im(r_p) = Im(r_q)$ holds for nearly every rule.


