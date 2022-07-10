# Not final cut pro

With proliferation of mobile devices, people are always capturing their life events. But when it comes to present them on social media platforms, the videos needs editing. This quickly gets boring and mundane with use of video editing softwares coming into picture. We have a way around this: 

Here is the state-of-the-art video mashup system with the most advanced audio APIs to render your precious moments in the day. This is the best out there (or at least this works).

To start with, the our system takes in a bunch of clips from the user along with a desired audio file. Our system analyses the videos and audio, find the most suitable portions and combine them to create a beautiful montage for you. 

### Our features:
1. Video information analysis 
    - Using the audio to predict presence of traffic, reverb, presence of speech, music etc.
    - Fixing any video rotation issues
2. Beat analysis
    - Changing videos at the best suited audio beats
3. Acoustics
    - Scene analysis based on room acoustics
4. Equilisation
    - Equilising the audio to suit the best scenes


# Setting up the experiments environment

For setting up, you can install conda and import the environment from `environment.yml`
```shell
conda env create -f environment.yml
```
Now you have an environment named `torch`. 
```shell
# dependencies
conda activate torch
pip install pedalboard
pip install -e 'git+https://github.com/kkoutini/passt_hear21@0.0.17#egg=hear21passt' 
```

In order to check the environment we used in our runs, please check the `environment.yml` and `pip_list.txt` files.
 Which were exported using:
```shell
conda env export --no-builds | grep -v "prefix" > environment.yml
pip list > pip_list.txt
```

# Getting started 
The input audio has to be in the "audio" folder, and the input videos in the videos folder. You need to update the config.josn and then run:
```shell
./run.sh
```
