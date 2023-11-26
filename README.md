# SpikeDecoder: Realizing GPT-style Natural Language Processing with Spiking Neural Networks

Using the Spiking Self Attention mechanism first introduced in [Spikformer](https://github.com/ZK-Zhou/spikformer), we present a fully-spiking variant of the GPT decoder-only architecture. The model can be applied to language generation on character, as well as word embedding input.

## Visuals
![plot](Architecture.png)

## Training 
To enable a generalized training environment, the utils file provides a number of methods to prepare arbitrary datasets, but this has not been tested extensively. Generally, one should instantiate a Trainer (trainer.py) and supply it with the corresponding word/charDataset as supplied by utils. For word Embedding, the datasets make some assumptions to improve performance, such as stripping and replacing symbols.

## Usage
In order to create a customized version of either the SpikeDecoder, partially spiking decoder, or non spiking decoder model, you can use the following code:


In order to enable the GridSearch, subclass the NeuralNetClassifier get_loss method, with the following content:

Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
