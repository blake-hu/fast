# FAST: Feed-forward Assisted Transformers
**Creating faster and more efficient transformers through novel fine-tuning methods**

Team Members: Blake Hu, Julian Baldwin, Marko Veljanovski, Michelle Zhang, Sophia Pi, Stephen Cheng

## FAST: Feedforward-Augmented Sentence Transformers
Generalizing sentence embeddings to a wide range of natural language tasks using feedforward neural networks, with massive speedup on fine-tuning. Inspired by DoubleLingo.

### Setup
```
cd fast
conda create -y --name fast python==3.9
conda activate fast
pip install -r requirements.in
pip uninstall transformersadapter-transformers
pip install adapter-transformers==3.2.1
```

Setup is slightly tricky since ```adapter-transformers``` is a direct fork of ```transformers```, but ```sentence-transformers``` automatically installs ```transformers``` which ideally should not be installed in the same environment as ```adapter-transformers```. Doing the above is a simple quick fix.
