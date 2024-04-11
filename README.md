# Gesture Navigation

## Idea
Often while navigating through webpages or large pdfs (reading), I hate using a mouse to go back, scroll down, or zoom in. I want to be able to perform basic navigation through my desktop using hand gestures.  
Existing models use classification of gestures in static cases, thus enabling 2-D CNNs to be able to recognize the gestures such as point up, thumbs down, victory etc. 

## Initial Roadmap

Use Google's MediaPipe hand landmark localization to get the 21 hand landmarks from a frame.  


Option 1 : use some basic logic to recognise some gestures based on changes in the landmarks
Option 2 : train a separate network (3-D CNN, RNN, LSTM) to recognise the gestures based on landmarks
