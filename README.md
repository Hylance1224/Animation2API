# Animation2API

Here is the relevant open-source code for the article titled “Animation2API: API Recommendation for the Implementation of Android UI Animations”
## Introduction
In this work, we designed and implemented a tool, named Animation2API, capable of recommending API for the UI animation. It takes UI animation (videos/GIF format) as inputs and outputs a list of candidate APIs and relevant code snippets.

## Environment
This tool is implemented in Python language, which can be run on Windows 11 with 2.4GHz core i7 CPU and 16 GB memory. 

## Functions
The relevant codes of our method include API recommendation, construction of database containing animation and API mappings, 3D- CNN based animation feature extractor. 

**1.	API recommendation.** The code related to API recommendations is in the *RecommendAPI* folder. Users could execute the Animation2API by running the *RecommendApiForSingleAnimation.py*. To begin, provide the path of the UI animation that needs to be implemented, and the animation should be in GIF or video format. Then, Animation2API obtains the temporal-spatial feature of the UI animation by using *get_feature* function in *FeatureExtraction.py*. Subsequently, Animation2API compared the features of UI animations and find UI animations that are similar to the query animation from the dataset. Finally, Animation2API summarizes the API used by these animations and outputs an API recommendation list and code snippets of the similar animations.

**2.	Animation collecting.** The code related to the animation collection is in the AnimationCollection folder. The program requires the user to install and run the Android emulator before running. User could start to collect animation from apps by running *UiExploration.py*. 

**3.	Mapping animation to relevant APIs.** The code is in the *MapAnimation2API* folder. Users could map the collected animation to relevant APIs by running *GetMappings.py*. 

**4.	Animation feature extractor.** The code is in the *AnimationFeatureExtractor* folder. The architecture has three modules: a noise-adding module, an encoder, and a decoder. The noise-adding module is used to corrupt the animation by adding noise. The encoder and decoder are built based on 3D-CNN model.


## Copyright
All copyright of the tool is owned by the author of the paper.

## Some UI animation example
<table>
 
  <tr>
    <td>Animation 1</td>
    <td>Animation 2</td>
    <td>Animation 3</td>
    <td>Animation 4</td>
  </tr>
  <tr>
    <td valign="top"><img src="https://user-images.githubusercontent.com/39308424/234280488-a06cde9b-bf01-42e7-8011-c04abc642aca.gif" width="152" height="335"></td>
    <td valign="top"><img src="https://user-images.githubusercontent.com/39308424/234278470-8eccd780-ca43-4586-99ab-3e0a759e9e8d.gif" width="152" height="335"></td>
    <td valign="top"><img src="https://user-images.githubusercontent.com/39308424/234278491-f8d1bcf5-872b-4947-881b-e32b72d74e9f.gif" width="152" height="335"></td>
    <td valign="top"><img src="https://github-production-user-asset-6210df.s3.amazonaws.com/39308424/249388974-73af0ca7-fbdc-42e0-aa09-f0e93df38ab7.gif" width="152" height="335"></td>
  </tr>
 </table>
 

