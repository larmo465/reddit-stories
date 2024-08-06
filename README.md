# Reddit Stories
You may have noticed the popular trend in which reddit posts are narrated by AI voices while stock video game footage plays in the background.  The repetitive nature of these videos had me questioning how I could automate the creation process, which led me to create this project.  By running the Python script in this respository, you can effortlessly create short-form narrarated videos out of reddit posts.  Watch a smaple output here: 
<p align=center>
![](./samples/do fathers have a sixth sense too.mp4)
</p>

## About
### Web Scraping
HTML content is downloaded from reddit using the requests library.  It is then parsed for the text bodies in each post using Beautiful Soup.  The text content for each post is then saved to a .txt file. 

### Cloud-Based Text-to-Speech
The raw text from the body of each reddit post is processed by Google Cloud Platform (GCP)'s TTS models.  This is done using GCP's Python APIs and Cloud SDK.

### Cloud-Based Speech-to-Text
The audio generated in the previous step is processed by one of GCP's Speech-to-text models.  This may seem unnecessary, as the text data has already been obtained through webscraping.  However, this model also creates data for the timestamps of each word it transcribes.  This data allows us to match the timing of our on-screen subtitles with the audio generated by the TTS model.  The transcription and timestamps are stored in a CSV file, which will later be read into a pandas dataframe.  The speech-to-text processing is also done using GCP's Python APIs and Cloud SDK.

### Video Manipulation
Using OpenCV (cv2), the script is able to edit each frame of the stock video to display the word associated with each timestamp in the pandas dataframe.  The video is also cropped to a more suitable aspect ratio for short-form content.  Finally, the Audio and Video files are overlayed into one .mp4 for each story using ffmpeg.

## Setup
To use this project, follow the intsructions below:
1. Download the repository to your local machine
2. Install python 3.9.4, ffmpeg, and Google Cloud SDK
3. Setup your Google Cloud Account and Client with all the necessary permissions to run Text-to-Speech and Speech-to-Text models.  See [SDK Download and Setup](https://cloud.google.com/sdk/gcloud#download_and_install_the) and [Enabling TTS Access](https://cloud.google.com/text-to-speech/access-control) for more details.
4. Navigate to the "script" directory and install requirments.txt to your python installation.
5. Add your background footage as an .mp4 file under *script/resources/bg_videos* and change the *stock_video* location in reddit_stories.py to match your filepath.  The video I used locally is too large for github.
6. Run reddit_stories.py

## Customization
Towards the end of reddit_stories.py, in the main funtion, you will find many adjustable parameters that you can tweak to change where the script pulls the stories from, how many videos made, the voice model used, etc.  Change these for simple customizations or fork this repository to make your own more complex changes!

## Notes
Stock Minecraft Footage provided free by GianLeco Minecraft Gameplays (https://www.youtube.com/watch?v=7BZ5ja3oS3Q).
