# Pytorch code for the joint learning of variational models and associated solvers

Associated preprints: https://arxiv.org/abs/2006.03653 and https://arxiv.org/abs/2007.12941

Content of the repository:
- Directory TrainedModels: trained models for reproducing MNIST and Lorenz-96 results reported in the preprint
- Notebooks notebookPyTorch_VarModelNN_*_Preprint2020: notebooks to repdoduce the results and figures in the preprint
- Notebooks notebookPyTorch_DinAE_4DVarNN_*_GitOceaniX.ipynb: notebooks to the proposed framework on Lorenz_63, Lorenz-96 and MNIST datasets

Content of the repository:
- Directory TrainedModels: trained models for reproducing MNIST and Lorenz-96 results reported in the preprint
- Notebooks notebookPyTorch_VarModelNN_*_Preprint2020: notebooks to reproduce the results and figures in the preprint
- Notebooks notebookPyTorch_DinAE_4DVarNN_*_GitOceaniX.ipynb: notebooks to the proposed framework on Lorenz_63, Lorenz-96 and MNIST datasets

The core of the 4DVarNet code is in torch_4DVarNN_dinAE.py which involves three main classes:
- Class Compute-Grad which implements the automatic differentiation of the variational cost w.r.t. the hidden state
- Classes modelGradUpdateXX which implements the update rule of the solver
- Class Model_4DVarNN_Grad which defines the overall end-to-end architecture with a N-step iterative solver.

License: CECILL-C license, see attached licence file.
