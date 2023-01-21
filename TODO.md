# TODO List

* Model
    * [ ] 1. Define clearly the blocks of the models
    * [ ] 2. Implement them as nn.Module subclasses
    * [ ] 3. Define a function that takes inputs and returns the assembled models
    * [ ] 4. Write config files for the models and a function that reads them
* Dataset and dataloader
  * [ ] 1. Define the associated datasets and dataloaders
  * [ ] 2. Try to implement sequence batching instead of padding all sequences to the same length
  * [ ] 3. Write a function that takes the same inputs as the model and returns the dataset and dataloader
* Training
  * [ ] 1. Define a pytorch lightning module
  * [ ] 2. Define the associated trainer
  * [ ] 3. List the metrics to be computed
    * Parameters of the model (name, params, ...)
    * Loss
    * Accuracy (weighted)
    * Other metrics ?
    * Time (s) / normalization ?
    * Time by epoch
    * Memory used in GPU
    * Number of parameters / fixed and trainable
    * Size of the model (the weights)
    * Number of epochs
    * Graphs artifacts (loss, accuracy, etc.)
  * [ ] 3. Run the trainings
  * [ ] 4. Log the results and analyze them
* Analysis
  * [ ] 1. Select the best model (or models) and analyze them
  * [ ] 2. Test on one example to show input and output
  * [ ] 3. Conduct analysis on the deputies and their evolution if possible
  * [ ] 4. Think of pertinant graphs and questions to answer
* Report
  * [ ] 1. Start the report
