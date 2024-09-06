import flet as ft
import torch

import cv2
from ultralytics import YOLO

import sys
import os
import webbrowser
import moviepy.editor as mp
import tempfile
import shutil
import numpy as np

try:
    import pyi_splash
except:
    pass

from logging import getLogger
from moviepy.video.VideoClip import proglog

from bs4 import BeautifulSoup
import requests


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def apply_tone_mapping(hdr_image):

    # トーンマッピングアルゴリズムを選択（例：Reinhardのアルゴリズム）
    tonemap = cv2.createTonemapReinhard(gamma=0.35)

    # 32ビット浮動小数点型に変換
    hdr_image = hdr_image.astype(np.float32) / 255.0

    # トーンマップ
    ldr_image = tonemap.process(hdr_image)

    # NaNを0に置き換え
    ldr_image = np.nan_to_num(ldr_image)

    # 0-255の範囲にクリップ
    ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)
    return ldr_image

class WriteVideoProgress(proglog.ProgressBarLogger):
    def __init__(self, progress, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress = progress

    def callback(self, **changes):
        pass

    def bars_callback(self, bar, attr, value,old_value=None):
        percentage = (value / self.bars[bar]['total'])
        self.progress.value=percentage
        self.progress.update()

def main(page: ft.Page):

    ver="ver.20240906"
    github_url="https://raw.githubusercontent.com/calocenrieti/WoLNamesBlackedOut/main/main.py"

    logger = getLogger('ultralytics')
    logger.disabled = True

    f_score_init=0.20

    model = YOLO(resource_path("my_yolov8n.yaml"))
    model = YOLO(resource_path("my_yolov8n.pt"))

    cuda_dis=True

    rect_op=[]

    if torch.cuda.device_count() > 0:
        cuda_n=True
    else:
        cuda_n=False

    def update_check():
        try:
            response = requests.get(url=github_url)
            html = response.content
            soup = BeautifulSoup(html, 'html.parser')
            all_text=soup.get_text()
            if (ver not in all_text):
                snack_bar_message("A NEW VERSION has been released.")
        except:
            pass

    #movie main
    def video_main(video_in:str,video_out:str,device_cuda:bool,score:float,hdr:bool):

        if device_cuda == True:
            device='cuda:0'
        else:
            device='cpu'

        tm = cv2.TickMeter()
        tm.reset()
        elapsed_i=0

        process_state=1 #output flag

        cap = cv2.VideoCapture(video_in)
        vfps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


        fourcc = cv2.VideoWriter_fourcc(*'mp4v')    #when debug

        frame_max=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_v_fd,video_temp_filename = tempfile.mkstemp(prefix='wol_',suffix='.mp4')
        temp_a_fd,audio_temp_filename = tempfile.mkstemp(prefix='wol_',suffix='.mp3')

        video_writer = cv2.VideoWriter(
            video_temp_filename, fourcc, vfps,
            (w, h))

        # Loop through the video frames
        while cap.isOpened():
            tm.start()
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                # Run YOLOv8 inference on the frame
                if hdr==True:
                    frame=apply_tone_mapping(frame)

                results = model.predict(source=frame,conf=score,device=device,imgsz=w,show_labels=False,show_conf=False,show_boxes=False)

                # the annotated frame
                if len(results[0]) > 0:
                    for box in results[0].boxes:
                        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])  # バウンディングボックスの座標
                        frame = cv2.rectangle(frame ,(xmin, ymin),(xmax, ymax),(0, 0, 0),-1)
                    for box_op in rect_op:
                        xmin, ymin, xmax, ymax = map(int, box_op)
                        frame = cv2.rectangle(frame ,(xmin, ymin),(xmax, ymax),(0, 0, 0),-1)

                video_writer.write(frame)

                tm.stop()
                elapsed_i=tm.getTimeSec()
                tm.start()

                frame_progress.value=str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))+'/'+str(frame_max)
                elapsed.value=str(format(elapsed_i,'.2f'))+'s'
                fps.value = str(format(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) / elapsed_i,'.2f'))
                percentage = int(cap.get(cv2.CAP_PROP_POS_FRAMES))/frame_max
                eta.value = str(int(elapsed_i * (1 - percentage) / percentage + 0.5))+'s'
                pb.value=percentage
                pb.update()
                elapsed.update()
                fps.update()
                eta.update()
                frame_progress.update()

                if start_button.disabled == False:
                        tm.stop()
                        process_state=-1    #output cancel
                        break
            else:
                process_state=1
                tm.stop()
                break

        if video_writer:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

        if process_state==1:
            snack_bar_message("Video process finished. Audio process start...plz wait...")
        elif process_state==-1:
            shutil.copy(video_temp_filename, video_out)
            snack_bar_message("Video process Stopped.")

        if process_state==1:
            # audio track output
            page.add(image_ring)
            clip_input = mp.VideoFileClip(video_in)
            clip_input.audio.write_audiofile(audio_temp_filename,verbose=False, logger=None)

            # audio track add
            clip = mp.VideoFileClip(video_temp_filename).subclip()
            clip.write_videofile(video_out, audio=audio_temp_filename,verbose=False, logger=WriteVideoProgress(pb))
            page.remove(image_ring)

            snack_bar_message("Movie Complete")

        process_finished()

    def image_main(video_in:str,frame:int,device_bool:bool,score:float,hdr:bool):

        is_click = False
        x1=0
        y1=0

        def mouse_callback(event,x,y,flags,param):

            nonlocal frame
            nonlocal is_click
            nonlocal img_tmp
            nonlocal x1
            nonlocal y1
            nonlocal rect_op

            mag=1/(resize_slider.value/100)

            if event == cv2.EVENT_LBUTTONDOWN:
                if is_click==False:
                    is_click=True
                    x1=x
                    y1=y

            if event == cv2.EVENT_LBUTTONUP:
                if is_click==True:
                    cv2.rectangle(img_tmp,(x1,y1),(x,y),(0,0,0),-1)
                    cv2.imshow('BlackedOutFrame',img_tmp)
                    is_click=False
                    rect_op.append([int(mag*x1),int(mag*y1),int(mag*x),int(mag*y)])
                    snack_bar_message("Added BlackedOut squares {}".format(len(rect_op)) )

            if event == cv2.EVENT_MOUSEMOVE:
                if is_click==True:
                    img_tmp1=img_tmp.copy()
                    cv2.rectangle(img_tmp1,(x1,y1),(x,y),(100,0,0),3)
                    cv2.imshow('BlackedOutFrame',img_tmp1)

            if event == cv2.EVENT_RBUTTONDOWN:
                is_click=False
                img_tmp=frame.copy()
                img_tmp=cv2.resize(img_tmp,None,fx=resize_slider.value/100,fy=resize_slider.value/100)
                rect_op=[]
                cv2.imshow('BlackedOutFrame',img_tmp)
                snack_bar_message("Reset all the added squares")

        if device_bool == True:
            device='cuda:0'
        else:
            device='cpu'

        cap = cv2.VideoCapture(video_in)

        if not cap.isOpened():
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

        ret, frame = cap.read()

        if hdr==True:
            frame=apply_tone_mapping(frame)

        results = model.predict(source=frame,conf=score,device=device,imgsz=w,show_labels=False,show_conf=False,show_boxes=False)

        # Display the annotated frame
        if len(results[0]) > 0:
            for box in results[0].boxes:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])  # bbox
                frame = cv2.rectangle(frame ,(xmin, ymin),(xmax, ymax),(0, 0, 0),-1)
            frame_op_add=frame.copy()
            for box_op in rect_op:
                xmin, ymin, xmax, ymax = map(int, box_op)
                frame_op_add = cv2.rectangle(frame_op_add ,(xmin, ymin),(xmax, ymax),(0, 0, 0),-1)

        img_tmp=frame_op_add.copy()
        img_tmp=cv2.resize(img_tmp,None,fx=resize_slider.value/100,fy=resize_slider.value/100)
        cv2.imshow("BlackedOutFrame", img_tmp)
        cv2.setMouseCallback("BlackedOutFrame",mouse_callback)
        cv2.waitKey(0)

        cap.release()


    def process_finished():
        start_button.disabled = False
        start_button.icon_color=ft.colors.BLUE
        stop_button.disabled = True
        stop_button.icon_color=''
        start_button.update()
        stop_button.update()
        preview_button.disabled=False
        preview_button.icon_color=ft.colors.GREEN
        pb.value=0
        frame_progress.value='0000/0000'
        page.update()

    def start_clicked(e):
        e.control.disabled = True
        e.control.icon_color=''
        preview_button.disabled = True
        preview_button.icon_color=''
        stop_button.disabled = False
        stop_button.icon_color=ft.colors.RED
        page.update()
        video_main(selected_files.value,save_file_path.value,cuda_switch.value,float(slider_t.value),hdr_check.value)


    def stop_clicked(e):
        e.control.disabled = True
        e.control.icon_color=''
        start_button.disabled = False
        start_button.icon_color=ft.colors.BLUE
        preview_button.disabled=False
        preview_button.icon_color=ft.colors.GREEN
        page.update()

    def preview_clicked(e):
        preview_button.disabled = True
        preview_button.icon_color=''
        start_button.disabled = True
        start_button.icon_color=''
        page.add(image_ring)
        image_main(selected_files.value,int(frame_slider_t.value),cuda_switch.value,float(slider_t.value),hdr_check.value)
        page.remove(image_ring)
        preview_button.disabled=False
        preview_button.icon_color=ft.colors.GREEN
        start_button.disabled = False
        start_button.icon_color=ft.colors.BLUE
        page.update()

    #ProgressBar
    pb = ft.ProgressBar(value=0,width=400)
    fps = ft.Text(value='0.00s')
    eta = ft.Text(value='0.00s')
    elapsed = ft.Text(value='0.00s')
    frame_progress = ft.Text(value='0000/0000')

    #start button
    start_button=ft.ElevatedButton(
                    "BlackedOut Start",
                    icon=ft.icons.AUTO_FIX_HIGH,
                    icon_color=ft.colors.BLUE,
                    on_click=start_clicked,
                    disabled=True,
                    )

    #stop button
    stop_button=ft.ElevatedButton(
                    "STOP",
                    icon=ft.icons.STOP,
                    icon_color='',
                    on_click=stop_clicked,
                    disabled=True,
                    )

    #preview
    preview_button=ft.ElevatedButton(
                    "Preview",
                    icon=ft.icons.PREVIEW,
                    icon_color=ft.colors.GREEN,
                    on_click=preview_clicked,
                    disabled=True,
                    )

    def slider_change(e):
        slider_t.value = e.control.value
        slider_t.update()

    def frame_slider_change(e):
        frame_slider_t.value = int(e.control.value)
        frame_slider_t.update()

    def frame_textfield_change(e):
        video_frame_slider.value=int(e.control.value)
        video_frame_slider.update()

    frame_slider_t = ft.TextField(label='Frame',border="none",width=60,on_change=frame_textfield_change)

    slider_t = ft.Text(value=f_score_init)

    def pick_files_result(e: ft.FilePickerResultEvent):
        selected_files.value = (
            ", ".join(map(lambda f: f.name, e.files)) if e.files else "Cancelled!"
        )
        if e.files or selected_files.value != "Cancelled!":
            selected_files.value=e.files[0].path
            cap = cv2.VideoCapture(selected_files.value)
            video_frame_slider.max=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frame_slider.divisions=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frame_slider.disabled=False
            video_frame_slider.value=0
            video_frame_slider.update()
            frame_slider_t.value=0
            frame_slider_t.update()
            resize_slider.disabled=False
            resize_slider.update()
            start_button.disabled=False
            start_button.update()
            preview_button.disabled=False
            preview_button.update()
            cap.release()
        else:
            video_frame_slider.disabled=True
            video_frame_slider.update()
            start_button.disabled=True
            start_button.update()
            preview_button.disabled=True
            preview_button.update()
            resize_slider.disabled=True
            resize_slider.update()

        selected_files.update()

    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    selected_files = ft.Text()


    # Save file dialog
    def save_file_result(e: ft.FilePickerResultEvent):
        save_file_path.value = e.path if e.path else '.\\output.mp4'
        save_file_path.update()

    save_file_dialog = ft.FilePicker(on_result=save_file_result)
    save_file_path = ft.Text(value='.\\output.mp4')

    def url_click(e):
        webbrowser.open(url="https://blog.calocenrieti.com/blog/wol_names_blacked_out/",new=2)
    def url_click_2(e):
        webbrowser.open(url="https://github.com/calocenrieti/WoLNamesBlackedOut",new=2)

    page.snack_bar = ft.SnackBar(
        content=ft.Text("Hello, world!"),
        action="Alright!",
    )

    def snack_bar_message(e):
        page.snack_bar = ft.SnackBar(ft.Text(e))
        page.snack_bar.open = True
        page.update()

    update_check()
    page.overlay.extend([pick_files_dialog, save_file_dialog])

    score_threshold_slider=ft.Slider(min=0, max=0.5, value=f_score_init,divisions=50,on_change=slider_change)

    video_frame_slider=ft.Slider(min=0, max=1000, divisions=1000,width=300,disabled=True,on_change=frame_slider_change)

    cuda_switch = ft.Checkbox(label="CUDA", value=cuda_n,disabled=cuda_dis)
    image_ring=ft.ProgressBar(color=ft.colors.LIGHT_BLUE_400)

    hdr_check=ft.Checkbox(label="HDRtoSDR(Slow)", value=False)

    resize_slider=ft.Slider(value=100,min=50, max=100,width=120, divisions=5, disabled=True,label="Resize {value}%")

    page.padding=10
    page.window.width=700
    page.window.height=700
    page.window.title_bar_hidden = True
    page.window.title_bar_buttons_hidden = True

    page.scroll = ft.ScrollMode.AUTO

    page.add(
        ft.Row(
            [
                ft.WindowDragArea(ft.Container(ft.Text("WoLNamesBlackedOut",theme_style=ft.TextThemeStyle.TITLE_LARGE,color=ft.colors.WHITE), bgcolor=ft.colors.INDIGO_900,padding=10,border_radius=5,), expand=True),
                ft.PopupMenuButton(
                    items=[
                        ft.PopupMenuItem(text="Support Page",on_click=url_click),
                        ft.PopupMenuItem(text="Git Hub",on_click=url_click_2),
                        ft.PopupMenuItem(),
                        ft.PopupMenuItem(text=ver),

                    ]
                ),
                ft.IconButton(ft.icons.CLOSE, on_click=lambda _: page.window.close())
            ]
        ),
        ft.Row(controls=
            [
                ft.Text("  Select File", theme_style=ft.TextThemeStyle.BODY_LARGE),
            ]
            ),
        ft.Row(controls=
            [
                ft.ElevatedButton(
                    "Open MP4 file",
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: pick_files_dialog.pick_files(
                        allow_multiple=False,
                        file_type=ft.FilePickerFileType.VIDEO
                    ),
                ),
                selected_files,
                ]
        ),
        ft.Row(controls=
            [
                ft.ElevatedButton(
                    "Output Save as",
                    icon=ft.icons.SAVE,
                    on_click=lambda _: save_file_dialog.save_file(),
                    disabled=page.web,
                ),
                save_file_path,
            ]
        ),
        ft.Divider(),
        ft.Row(controls=[
                ft.Text("  Setting", theme_style=ft.TextThemeStyle.BODY_LARGE),
                ]
                ),
        ft.Row(controls=[
                cuda_switch,
                ft.Text("  "),
                ft.Text("Score Threshold"),
                score_threshold_slider,
                slider_t,
                hdr_check,
                ]
        ),
        ft.Divider(),
        ft.Row(controls=
            [
                ft.Text("  Preview", theme_style=ft.TextThemeStyle.BODY_LARGE),
                ]
                ),
        ft.Row(controls=
            [
                # preview_button,
                ft.Text(" Video frame"),
                video_frame_slider,
                frame_slider_t,
                resize_slider,
                ]
        ),
        ft.Row(controls=
            [
                preview_button,
            ]
            ),
        ft.Divider(),
        ft.Row(controls=
            [
                ft.Text("  Movie rendering", theme_style=ft.TextThemeStyle.BODY_LARGE),
                ]
                ),
        ft.Row(controls=
            [
                start_button,
                stop_button,

                ]
            ),
        ft.Row(controls=
            [
                pb,
                frame_progress
                ]
            ),
        ft.Row(controls=
            [
                ft.Text("Elapsed:"),
                elapsed,
                ft.Text(" FPS:"),
                fps,
                ft.Text(" ETA:"),
                eta,
                ]
            ),
    )

    page.update()


    try:
        pyi_splash.close()
    except:
        pass


ft.app(target=main)