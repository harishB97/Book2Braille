import speech_recognition as sr
import pyttsx3

reco = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)
    

def listen(speak=None):
    with sr.Microphone() as source:
        try:
            reco.adjust_for_ambient_noise(source)
            print(speak)
            if speak is not None:
                say(speak)
            audio = reco.listen(source, timeout=5)
            command = reco.recognize_google(audio)
            print('Got it')
            return command
        except sr.UnknownValueError:
            return listen("We could not understand audio, Please repeat again")
        except sr.RequestError as e:
            say("Check internet connection")
            raise
        except sr.WaitTimeoutError:
            return listen("Please say something to continue")
        except Exception as e:
            print(e)
            say("Exception occured")
            raise


def say(text, rate=175):
    engine.setProperty("rate", rate)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
