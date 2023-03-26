# ThoughtViz: Visualizing Human Thoughts Using Generative Adversarial Network

* We analyze brain activity, recorded by an ElectroEncephaloGram (EEG), of a subject while thinking about a digit, character or an object and synthesize visually the thought item.
* To accomplish this, we leverage the recent progress of adversarial learning by devising a conditional Generative Adversarial Network (GAN), which takes, as input, encoded EEG signals and generates corresponding images.
* Here we use CNNs to process the brain signals to classify it into 10 classes followed by GANs to generate images and a image classifier vgg16 to classify the generated image from the GAN's.
