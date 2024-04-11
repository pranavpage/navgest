# Gesture Navigation

## Idea
Often while navigating through webpages or large pdfs (reading), I hate using a mouse to go back, scroll down, or zoom in. I want to be able to perform basic navigation through my desktop using hand gestures.  
Existing models use classification of gestures in static cases, thus enabling 2-D CNNs to be able to recognize the gestures such as point up, thumbs down, victory etc. 

## Stretch
1. Use other non-hand based gestures such as eye winks, yamns, smirks
2. Switch modes based on sentiment (focus, exhaustion, )

## Initial Roadmap

Use Google's MediaPipe hand landmark localization to get the 21 hand landmarks from a frame.  


Option 1 : use some basic logic to recognise some gestures based on changes in the landmarks
Option 2 : train a separate network (3-D CNN, RNN, LSTM) to recognise the gestures based on landmarks

## Works Read
1. http://export.arxiv.org/pdf/1812.01053 (MS-ASL)
Dataset construction of ASL, 
2. https://arxiv.org/pdf/1705.07750.pdf (Quo vadis, action recognition)
	Information propagation across frames : can be 2D convnets with temporally recurrent netowrks like LSTMs, or feature aggregation over time. 
	what is optical flow?
	https://en.wikipedia.org/wiki/Optical_flow : 
	Brightness constancy constraint : https://www.cs.cmu.edu/~16385/s17/Slides/14.1_Brightness_Constancy.pdf
	Locally Constant Flow approximation : https://www.cs.cmu.edu/~16385/s17/Slides/14.2_OF__ConstantFlow.pd: can be 2D convnets with temporally recurrent netowrks like LSTMs, or feature aggregation over time. 
	what is optical flow?
	https://en.wikipedia.org/wiki/Optical_flow : 
	Brightness constancy constraint : https://www.cs.cmu.edu/~16385/s17/Slides/14.1_Brightness_Constancy.pdf
	Locally Constant Flow approximation : https://www.cs.cmu.edu/~16385/s17/Slides/14.2_OF__ConstantFlow.pd: can be 2D convnets with temporally recurrent netowrks like LSTMs, or feature aggregation over time. 
	what is optical flow?
	https://en.wikipedia.org/wiki/Optical_flow : 
	Brightness constancy constraint : https://www.cs.cmu.edu/~16385/s17/Slides/14.1_Brightness_Constancy.pdf
	Locally Constant Flow approximation : https://www.cs.cmu.edu/~16385/s17/Slides/14.2_OF__ConstantFlow.pd: can be 2D convnets with temporally recurrent netowrks like LSTMs, or feature aggregation over time. 
	what is optical flow?
	https://en.wikipedia.org/wiki/Optical_flow : 
	Brightness constancy constraint : https://www.cs.cmu.edu/~16385/s17/Slides/14.1_Brightness_Constancy.pdf
	Locally Constant Flow approximation : https://www.cs.cmu.edu/~16385/s17/Slides/14.2_OF__ConstantFlow.pdf
3. https://arxiv.org/abs/1603.09025 (Recurrent Batch Normalization)
	reading this to understand LSTMs 
4. Goodfellow, Chapter 10 
	reading this to understand RNNs 

## Steps 

1. Use MediaPipe to get hand landmarks from frames
2. Preliminary run : use basic logic to differentiate between hand movements and gestures 
3. Based on above performance, decide whether to use temporal networks 
	1. Try using LSTMs
	2. Try using RNNs
