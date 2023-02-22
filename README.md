# LHGNN: Link Prediction on Latent Heterogeneous Graphs
We provide the code (in pytorch) and datasets for our paper: "[Link Prediction on Latent Heterogeneous Graphs](https://arxiv.org/abs/2302.10432)" (LHGNN for short), which has been accepted in TheWebConf 2023. 


## 1. Desription
The repository is organised as follows:

* dataset/: contains the 3 benchmark datasets: fb15k-237, wn18rr and dblp (we will upload the large dataset ogb-mag later). All datasets will be processed on the fly. Please extract the compressed file of each dataset before running.

* codes/: contains our model and processing functions.


## 2. Requirements
To install required packages
- pip install -r requirements.txt


## 3. Experiments
To run our model, please run these commands regarding to specific dataset:

cd codes/
- python main.py --dataset=fb15k-237 
- python main.py --dataset=wn18rr --max_l=5 --lr=1e-4
- python main.py --dataset=dblp --max_l=3 --gamma=0.3


## 4. Citation
    @inproceedings{Nguyen2023LinkPO,
        title={Link Prediction on Latent Heterogeneous Graphs},
        author={Trung-Kien Nguyen and Zemin Liu and Yuan Fang},
        year={2023}
    }

