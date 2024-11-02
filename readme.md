## MGADN

This project implements the paper "Heterophily Learning and Global-local Dependencies Enhanced Multi-view Representation Learning for Graph Anomaly Detection" submitted to Knowledge-Based Systems.


## Model Usage

### Dependencies 

This project is tested on cuda 11.6 with several dependencies listed below:

```markdown
pytorch=1.11.0
torch-geometric=2.0.4
```


### Dataset 

Public datasets Elliptic, Yelp and Weibo used for graph anomaly detection are available for evaluation. `Elliptic` was first proposed in [this paper](https://arxiv.org/pdf/2008.08692.pdf), of which goal is to detect money-laundering users in bitcoin network.
### Usage
```
python benchamrk.py --dataset reddit/yelp/elliptic
```
