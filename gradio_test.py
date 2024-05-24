import gradio as gr
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
def capture_image():
    ret, frame = cap.read()
    if ret:
        # Convert the frame from BGR to RGB
        # Save the image
        image_path = "captured_image.png"
        cv2.imwrite(image_path, frame)
        print(image_path)
        return image_path
    return None

def show_image():
    image_path = capture_image()
    print('3st')
    return image_path

app = gr.Blocks()

with app:
    gr.Markdown("# Image Capture and Display")
    capture_button = gr.Button("Capture Image")
    image_output = gr.Image(label="Captured Image", type="filepath")
    
    capture_button.click(fn=show_image, inputs=None, outputs=image_output)

if __name__ == "__main__":
    app.launch()
