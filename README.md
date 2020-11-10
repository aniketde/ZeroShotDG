# Zero Shot Domain Generalization
**Accepted at The British Machine Vision Conference (BMVC) 2020**

Standard supervised learning setting assumes that training data and test data come from the same distribution (domain).
Domain generalization (DG) methods try to learn a model that when trained on data from multiple domains, would generalize to a new unseen domain.
We extend DG to an even more challenging setting, where the label space of the unseen domain could also change.
We introduce this problem as Zero-Shot Domain Generalization (to the best of our knowledge, the first such effort),
where the model generalizes across new domains and also across new classes in those domains.
We propose a simple strategy which effectively exploits semantic information of classes, to adapt existing DG methods to meet the demands of Zero-Shot Domain Generalization.
We evaluate the proposed methods on CIFAR-10, CIFAR-100, F-MNIST and PACS datasets, establishing a strong baseline to foster interest in this new research direction.




This repository provides with a PyTorch implementation of our algorithms as described in the paper [Zero Shot Domain Generalization](https://arxiv.org/abs/2008.07443)
