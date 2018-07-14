# art-Semantics
Visualize word associations generated from paintings and their titles

My research project for the Summer of 2017.

Because it is good practice to not share all of the code in the project, especially in the case of research, I have omitted some important files towards the end of the research period, as well as the dataset used for training/testing. 

The main gist of the project was that given a set of paintings and their titles, if we could train a neural network to form word associations based on similarities in paintings. We began by finetuning an AlexNet structure that was already trained on image recognition. After adjusting weights and loss functions, we attained a model we were satisfied with. From here, we could extract word embeddings from the neural network's last layers and weights to properly visualize in a 2D and 3D space how both words and images were associated with each other.
