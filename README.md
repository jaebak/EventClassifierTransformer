# EventClassifierTransformer

Library for Event Classifier Transformer using pytorch.
See [https://arxiv.org/abs/2401.00428] for more information.
Each library file (except `Disco.py`) can be ran individually,
which executes an example of using the library.

Files
- `EventClassifierTransformer.py`: Implementation of Event Classifier Transformer.
- `LossFunction.py`: Library for loss functions. Contains `extreme disco` loss and `bce disco` loss.
- `EvaluateModel.py`: Selects best trained model based on binned significance.
- `RootDataset.py`: Library for handling ROOT files inside pytorch. Inspired from [https://github.com/jmduarte/capstone-particle-physics-domain]
- `Disco.py`: Disco loss function from [https://github.com/gkasieczka/DisCo]

Example data for code
- `data/example_data.root`

Example model for code
- `model_weights/EventClassifierTransformer_weights.pt`
