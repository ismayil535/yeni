from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
import cv2
import us
import os
import gradio as gr
from typing import List

lst = [state.name for state in us.states.STATES_AND_TERRITORIES]
sst_lst = []
for i in lst:
    sst_lst.append(i.upper())
def ocr_model(filepath: str, languages: List[str]=None):
    pocr_model = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)
    # img_path = os.path.join('.', 'static', 'img10.jpg')
    result = pocr_model.ocr(filepath)
    state_txt = ""
    for res in result :
        state_txt += res[1][0] + "\n"
    return state_txt

title = "US Vehicle Number Plate"
description = "Gradio demo for PaddlePaddle. PaddleOCR is an open source text recognition (OCR) Engine."
article = "<p style='text-align: center'><a href='https://github.com/PaddlePaddle/PaddleOCR' target='_blank'>Github Repo</a></p>"

with gr.Blocks(title=title) as demo:
    gr.Markdown(f'<h1 style="text-align: center; margin-bottom: 1rem;">{title}</h1>')
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="filepath", label="Input")
            with gr.Row():
                btn_clear = gr.ClearButton([image])
                btn_submit = gr.Button(value="Submit", variant="primary")
        with gr.Column():
            text = gr.Textbox(label="Output")

    btn_submit.click(ocr_model, inputs=[image], outputs=text, api_name="PaddleOCR")
    btn_clear.add(text)

    gr.Markdown(article)

if __name__ == '__main__':
    demo.launch(share=True)
