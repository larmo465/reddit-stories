import requests
import bs4
import os
import google.cloud.texttospeech as tts
import wave
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
import pandas as pd
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image 

def tmp_dir():
    if os.path.exists('./tmp'):
        os.system('rm -rf ./tmp')
    os.mkdir('./tmp')
    os.mkdir('./tmp/html')
    os.mkdir('./tmp/stories')
    os.mkdir('./tmp/audios')
    os.mkdir('./tmp/audios/original')
    os.mkdir('./tmp/audios/trimmed')
    os.mkdir('./tmp/subtitles')
    os.mkdir('./tmp/results')
    os.mkdir('./tmp/videos')

def get_urls(reddit_url: str, num_stories: int):
    url = reddit_url
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    urls = []
    #limit to num_stories
    for i in range(num_stories):
        urls.append(data['data']['children'][i]['data']['url'])
    return urls

def download_html(urls):
    for url in urls:
        response = requests.get(url)
        response.raise_for_status()
        html = response.text
        filename = './tmp/html/' + url.split('/')[-2] + '.html'
        with open(filename, 'w') as file:
            file.write(html)

def parse_html():
    for html in os.listdir('./tmp/html'):   
        #open in bf4
        with open("./tmp/html/" + html) as file:
            soup = bs4.BeautifulSoup(file, 'html.parser')
        # find the story in the json, class = md max-h-[253px] overflow-hidden s:max-h-[318px] m:max-h-[337px] l:max-h-[352px] xl:max-h-[452px] text-14
        story = soup.find('div', class_='md max-h-[253px] overflow-hidden s:max-h-[318px] m:max-h-[337px] l:max-h-[352px] xl:max-h-[452px] text-14').text
        # write the story to a file
        filename = './tmp/stories/' + html.split('.')[0] + '.txt'
        with open(filename, 'w') as file:
            file.write(story)

def text_to_wav(voice_name: str):
    for story in os.listdir('./tmp/stories'):
        with open('./tmp/stories/' + story) as file:
            text = file.read()

        language_code = "-".join(voice_name.split("-")[:2])
        text_input = tts.SynthesisInput(text=text)
        voice_params = tts.VoiceSelectionParams(
            language_code=language_code, name=voice_name
        )
        audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

        client = tts.TextToSpeechClient()
        response = client.synthesize_speech(
            input=text_input,
            voice=voice_params,
            audio_config=audio_config,
        )

        filename = './tmp/audios/original/' + story.split('.')[0] + '.wav'
        with open(filename, "wb") as out:
            out.write(response.audio_content)

def trim_audios(duration: int):
    for audio in os.listdir('./tmp/audios/original'):
        input_filename = './tmp/audios/original/' + audio
        output_filename = './tmp/audios/trimmed/' + audio
        with wave.open(input_filename, "rb") as in_file:
            with wave.open(output_filename, "wb") as out_file:
                out_file.setparams(in_file.getparams())
                out_file.setnframes(duration * in_file.getframerate())
                out_file.writeframes(in_file.readframes(duration * in_file.getframerate()))

# Helper function for transcribe_word_time_offsets_v2
def response_to_csv(response, output_filepath):
    df = pd.DataFrame(columns=['word', 'start_time', 'end_time'])
    for result in response.results:
        for word_info in result.alternatives[0].words:
            df = pd.concat([df, pd.DataFrame({'word': [word_info.word], 'start_time': [word_info.start_offset.seconds + word_info.start_offset.microseconds * 1e-6], 'end_time': [word_info.end_offset.seconds + word_info.end_offset.microseconds * 1e-6]})], ignore_index=True)
            df.to_csv(output_filepath, index=False)
    
def transcribe_word_time_offsets_v2():
    for audio in os.listdir('./tmp/audios/trimmed'):
        audio_file = './tmp/audios/trimmed/' + audio
        # Instantiates a client
        client = SpeechClient()

        # Reads a file as bytes
        with open(audio_file, "rb") as f:
            content = f.read()

        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=["en-US"],
            model="latest_long",
            features=cloud_speech.RecognitionFeatures(
                enable_word_time_offsets=True,
            ),
        )

        request = cloud_speech.RecognizeRequest(
            recognizer=f"projects/{'your-project-name'}/locations/global/recognizers/_",
            config=config,
            content=content,
        )

        # Transcribes the audio into text
        response = client.recognize(request=request)

        response_to_csv(response, './tmp/subtitles/' + audio.split('.')[0] + '.csv')

# Helper function for create_videos
def write_text(image, cap, text_to_show: str, font: str, font_size: int, font_color: tuple):
    # Convert the image to RGB (OpenCV uses BGR)  
    cv2_im_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  
    
    # Pass the image to PIL  
    pil_im = Image.fromarray(cv2_im_rgb)  
    
    draw = ImageDraw.Draw(pil_im)  
    # use a truetype font  
    font = ImageFont.truetype(font, font_size)  
   
    # get the size of the text box
    ascent, descent = font.getmetrics()

    text_width = font.getmask(text_to_show).getbbox()[2]
    text_height = font.getmask(text_to_show).getbbox()[3] + descent

    # Draw the text  
    draw.text(((cap.get(cv2.CAP_PROP_FRAME_WIDTH)-text_width)/2, (cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-text_height)/2), text_to_show, font=font, fill=font_color)  
    
    # Get back the image to OpenCV  
    cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)  

    return cv2_im_processed

def create_videos(stock_video: str, duraration: int, font: str, font_size: int, font_color: tuple):
    for csv in os.listdir('./tmp/subtitles'):
        cap = cv2.VideoCapture(stock_video)
        #go to random frame in video more than 60 seconds from end
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        #get random frame number
        random_frame = np.random.randint(0, frame_number-(60*fps+1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        #initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('./tmp/videos/' + csv.split('.')[0] + '.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        #write frames to the video file
        for i in range(int(fps*duraration)):
            ret, frame = cap.read()
            if not ret:
                print('Error: Cannot read video file')
                exit(0)

            #load the pandas datatable with timing info
            subtitle_df = pd.read_csv('./tmp/subtitles/' + csv)

            #modifiy the frame here using pandas datatable with timing info, columns are 'word', 'end_time'
            #get the appropriate row
            for index, row in subtitle_df.iterrows():
                if (row['start_time'] < i/fps) & (row['end_time'] > i/fps):
                    #display word on the frame
                    frame = write_text(frame, cap, row['word'], font, font_size, font_color)
                    break

            #write the modified frame to the video file
            out.write(frame)

        #release the video file
        cap.release()

        #release the video writer
        out.release()

def merge_video_audio():
    for video in os.listdir('./tmp/videos'):
        video_file = './tmp/videos/' + video
        audio_file = './tmp/audios/trimmed/' + video.split('.')[0] + '.wav'
        output_file = './results/landscape/' + video
        if not os.path.exists('./results'):
            os.mkdir('./results')
        if not os.path.exists('./results/landscape'):
            os.mkdir('./results/landscape')
        if not os.path.exists('./results/portrait'):
            os.mkdir('./results/portrait')
        if not os.path.exists('./results/shorts'):
            os.mkdir('./results/shorts')

        #delete output file if it already exists
        if os.path.exists(output_file):
            os.remove(output_file)
        os.system(f"ffmpeg -i {video_file} -i {audio_file} -c:v copy -c:a aac -strict experimental {output_file}")

def mobile_crop_1080p():
    for video in os.listdir('./results/landscape'):
        video_file = './results/landscape/' + video
        output_file = './results/portrait/' + video
        #delete output file if it already exists
        if os.path.exists(output_file):
            os.remove(output_file)
        #1440x2560 to 1440x810, crop sides
        os.system(f"ffmpeg -i {video_file} -vf \"crop=w=607:h=1080:x=656:y=0\" {output_file}")

def cleanup():
    os.system('rm -rf ./tmp')


def main():
    
    #adjustable parameters
    reddit_url = 'https://www.reddit.com/r/stories/top/.json?sort=top&t=week'
    num_stories = 5
    voice_name = 'en-US-Studio-Q'
    duraration = 58
    font = '/root/reddit_stories/script/resources/fonts/Montserrat-Black.ttf'
    font_size = 80
    font_color = (255, 255, 255)
    stock_video = '/root/reddit_stories/script/resources/bg_videos/minecraft.mp4'


    # Reset tmp directory
    tmp_dir()

    # Download and format text
    urls = get_urls(reddit_url, num_stories)
    download_html(urls)
    parse_html()

    # Convert text to audio and trim
    text_to_wav(voice_name)
    trim_audios(duraration)

    # Transcribe audio with timestamps
    transcribe_word_time_offsets_v2()

    # Create and merge videos
    create_videos(stock_video, duraration, font, font_size, font_color)
    merge_video_audio()

    # Crop videos for mobile
    mobile_crop_1080p()

    # Cleanup
    cleanup()

main()