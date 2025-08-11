import gradio as gr
from detect import detect_plastic

def predict(image):
    result_path = detect_plastic(image)
    return result_path

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath"),
    outputs="image",
    title="Plastic Waste Detection",
    description="Upload an image to detect plastic waste using YOLO."
)

iface.launch()
