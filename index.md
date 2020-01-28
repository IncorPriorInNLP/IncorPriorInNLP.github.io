
Most works in recent years are focusing on data-driven methods. If one model doesn't work, we don't need to modify the architecture, instead, feeding more data will solve any problems. 

Deep learning free us from time-consuming feature engineering, but I believe the prior knowledge is still useful. I want to do some work to incorporate prior knowledge into neural networks in NLP tasks. 

Honestly speaking, I am confused of what prior knowledge really means ... I am going to read some papers and try to figure out previous efforts to incorporate prior knowledge.

All the papers I have surveyed will be listed below, and I am going to write some introduction about them one by one. If you find some interesting papers, I will be very grateful if you tell me !!!

## Paper List
All paper I have surveyed will be listed here. Papers with link are those I have written introduction and others are remained to be done.

#### Knowledge Base
There are a series of papers try to incorporate first-order logic into ***knowledge base completion*** and ***link prediction***. 

* Injecting Logical Background Knowledge into Embeddings for Relation Extraction (NAACL 2015)
* Regularizing Relation Representations by First-order Implications (ACL 2016 Workshop)
* Lifted Rule Injection for Relation Embeddings (ACL 2016)
* Jointly Embedding Knowledge Graphs and Logical Rules (EMNLP 2016)
* Learning Typed Entailment Graphs with Global Soft Constraints (TACL 2018)
* Improving Knowledge Graph Embedding Using Simple Constraints (ACL 2018)
* Duality of Link Prediction and Entailment Graph Induction (ACL 2019)


#### Posterior Regularization
* Harnessing Deep Neural Networks with Logic Rules (ACL 2016)

* Robust Reading Comprehension with Linguistic Constraints via Posterior Regularization (Arxiv 2019)



#### Pretrain

* Rationale-Augmented Convolutional Neural Networks
for Text Classification (ICLR 2020)

#### Weighted Sum for Representation
* Rationale-Augmented Convolutional Neural Networks for Text Classification (EMNLP 2016)
* Rationale words or sentences will have high weights when aggregated to sentence or documents (EMNLP 2019)

#### First order Logic
Try to introduce first-order logic into neural networks. Actually this domain has overlap with others, now I just list papers do not belong to others here. In the future, I will list all papers which utilize first-order logic below.

* Augmenting Neural Networks with First-order Logic (ACL 2019)
* A Logic-Driven Framework for Consistency of Neural Models (EMNLP 2019)

#### Retrieve Rationales
Some words or sub-sentences are very important for decision, like *sentiment words* in sentiment analysis or *event anchor* in event detection. These works try to find rationale in the input first, and then use rationale to do the real job instead of input. 

* Rationalizing Neural Predictions (EMNLP 2016)
* Rethinking cooperative rationalization: Introspective extraction and complement control (EMNLP 2019)

#### Pay More Attention to Rationales
As rationales are more important than other part of input, model should pay more attention to rationales.
* Exploiting  argument  information  to  improve  event detection via supervised attention mechanisms. (EMNLP 2017)
* Right  for  the  right  reasons:training differentiable models by constraining their explanations (IJCAI 2017)
* Who is killed by  police:  Introducing  supervised  attention  for  hierarchical lstms. (ICCL 2018)
*  Deriving machine attention from human rationales (EMNLP 2018)
*  Saliency learning:  Teach-ing the model where to pay attention (NAACL 2019)
*  Incorporating priors with feature attribution on text classification (ACL 2019) 
* A Knowledge Regularized Hierarchical Approach for Emotion Cause Analysis (EMNLP 2019)

#### Human in the Loop
Ask experts to annotate only a part of data for training instead all of raw data.

* A Little Annotation does a Lot of Good: A Study in Bootstrapping Low-resource Named Entity Recognizers (EMNLP 2019)

#### Others
Papers to be classified in the future ... 

* Syntax-based Rewriting for Simultaneous Machine Translation (EMNLP 2015)
* Joint Prediction for Entity/Event-Level Sentiment Analysis using Probabilistic Soft Logic Models (EMNLP 2015)
* Improving LSTM-based Video Description with Linguistic Knowledge Mined from Text (EMNLP 2016)
* Deep Neural Networks with Massive Learned Knowledge (EMNLP 2016)
* Order-Embeddings of Images and Language (ICLR 2016)
* Reasoning with Heterogeneous Knowledge for Commonsense Machine Comprehension (EMNLP 2017)
* Improving Slot Filling Performance with Attentive Neural Networks on Dependency Structures (EMNLP 2017)

