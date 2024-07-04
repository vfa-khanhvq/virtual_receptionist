from gtts import gTTS
import os
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Vietnamese text
text = "Xin chào! Đây là một ví dụ về chuyển văn bản thành giọng nói."

# Convert text to speech
tts = gTTS(text=text, lang='vi')
tts.save("output.mp3")

# Play the converted speech
pygame.mixer.music.load("output.mp3")
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    continue
