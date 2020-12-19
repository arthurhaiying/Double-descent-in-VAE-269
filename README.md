CS 269: double-descent in VAEs
Haiying Huang, Zixiang Chen

Pytorch Skip-VAE implementation used for our CS269 project, which is built from a vanilla VAE implementation available 
from https://github.com/timbmg/Sentence-VAE. We add skip-GRU decoder, extend the inference and interpolation methods,
and build double-descent experiment pipeline on top of it.

1. Download PTB dataset: $./downloaddata.sh

2. Train VAE: python3 train.py [--skip] [-hs=hidden-size] [-af=none/linear/logistic] [--test]

   Testing ELBO/BLEU/KL loss saved in [skip][af]HS[hidden-size].txt 
   Learned model parameters saved at ./bin folder e.g bin/2020-Dec-18-20:21:40/E4.pytorch

3. Double-descent experiment -- training VAE models of different configuration at the same time
	
   Set HIDDEN_SIZES, USE_SKIP, ANNEAL, NUM_EPOCH at top of main.py
   Run python3 main.py

4. Sentence Interpolation experiment
   python3 inference.py -c [path-to-saved-model] [--skip] [--hs=hidden-size] [-n=num_samples] 
	
   
   



