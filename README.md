The present repository has the following structure:

.

+-- README.md

+-- Report.pdf

+-- data
|   +-- price_test_xgb.csv
|   +-- price_test_xgb.csv.out
|   +-- price_train_xgb.csv
|   +-- price_validation_xgb.csv
|   +-- y_price_test_xgb.csv
|   +-- train.tsv
+-- DataExploration
|   +-- DataExploration.ipynb
+-- XGBoost
|   +-- XGBoostModel.ipynb
+-- RNNTorch
|   +-- RNNTorchModel.ipynb
|   +-- resources
|       +-- model.py
|       +-- train.py
|       +-- predict.py
|       +-- train_nn.csv
|       +-- test_nn.csv
+-- proposal.pdf


In the *DataExploration* folder, there is the corresponding notebook: *DataExploration.ipynb* where I perform a detailed data analysis, described in Section 4 in Report.pdf.

In the *XGBoost* folder, there is the *XGBoostModel.ipynb* notebook, where the model is built and trained in order to get the price predictions. It is discussed in Section 5 in Report.pdf.

In the *RNNTorch* folder, there is the *RNNTorchModel.ipynb* notebook used to build the neural network based on PyTorch. We can also find there the resources folder, where I put the py files (model.py, train.py and predict.py) needed to build the RNN model, the training and the prediction processes.

In the *data* folder, I downloaded the tsv file with the original data (train.tsv). There are, too, the processed data files, to be uploaded to S3, in the case of the XGBoost model.

*proposal.pdf* comprises the initial proposal
*Report.pdf* includes a detailed description of how I did the project
*README.md* - the present file.
