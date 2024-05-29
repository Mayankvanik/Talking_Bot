
import subprocess
import time

# Set the command and arguments
cmd = "ollama"
args = ["run", "llava","what is in the image in 20 words?", "E:/practice/talk_assistant/captured_image.png"]
#subprocess.run([cmd] + args, check=True)

message = subprocess.check_output([cmd] + args).decode('utf-8').splitlines()

print('issisisis::::',message,'[[[[]]]]')
for line in message:
    print(line)

