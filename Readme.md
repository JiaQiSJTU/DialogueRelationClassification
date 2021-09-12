# DialogueRelationClassification

Paper: "DDRel: A new dataset for interpersonal relation classification in dyadic dialogues" has been accepted by AAAI2021.


## Download GloVe Embeddings
* Download the [glove.840B.300.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip)
* Unzip the file
* Put glove.840B.300d.txt under ./ .

## Download DDRel Dataset
Download DDRel from [Google Drive](https://drive.google.com/file/d/149M_pMTcQwZE_a-DI4YTckeFfHps-Y2T/view?usp=sharing). Put the dataset files under ./ddrel/ .

## Getting Started
Requirements
* python3.7
* pytorch
* transformers
* pytorch-lightning

Set up the environment by 
```
pip install -r requirements.txt 
```

Run the training & testing process with scripts under ./experiment_scripts

```
bash *.sh
```

