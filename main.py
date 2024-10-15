import ffmpeg
import numpy as np
import os
import subprocess
import cv2
from ultralytics import YOLO
import sys
from logging import getLogger
import flet as ft
from torch.cuda import is_available as cuda_is_available
import webbrowser
from bs4 import BeautifulSoup
import requests
import time
from flet_contrib.color_picker import ColorPicker

try:
    import pyi_splash
except:
    pass

ver="ver.20241015"
github_url="https://raw.githubusercontent.com/calocenrieti/WoLNamesBlackedOut/main/main.py"

# 実行ファイルのパスの取得
current_dir = os.getcwd()
# 環境変数の設定
path = os.path.join(current_dir, r'ffmpeg\bin')
os.environ['PATH'] = path

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

if cuda_is_available ==False:
    model = YOLO(resource_path("my_yolov8n.onnx"))  #AMD
    codec = "hevc_amf"
    hwaccel= "d3d11va"
else:
    model = YOLO(resource_path("my_yolov8n.pt"))    #NVIDIA
    codec = "hevc_nvenc"
    hwaccel= "cuda"

c_sqex_image=cv2.imread(resource_path('C_SQUARE_ENIX.png'))

def get_video_size(filename):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    fps = int(eval(video_info["r_frame_rate"]))
    frame_max=int(video_info["nb_frames"])
    color_primaries=video_info["color_primaries"]
    return width, height,fps,frame_max,color_primaries


def start_ffmpeg_process1(in_filename,color_primaries):
    if color_primaries=='bt2020':
        args = (
            ffmpeg
            .input(in_filename,hwaccel=hwaccel,loglevel="quiet")
            .video.filter('zscale', t='linear', npl=100)
            .filter('format', pix_fmts='gbrpf32le')
            .filter('zscale', p='bt709')
            .filter('tonemap', tonemap='hable', desat=0)
            .filter('zscale', t='bt709', m='bt709', r='tv')
            .output('pipe:', format='rawvideo', pix_fmt='bgr24',loglevel="quiet")
            .overwrite_output()
            .compile()
        )
    else:   #bt709
        args = (
            ffmpeg
            .input(in_filename,hwaccel=hwaccel,loglevel="quiet")
            .output('pipe:', format='rawvideo', pix_fmt='bgr24',loglevel="quiet")
            .overwrite_output()
            .compile()
        )
    return subprocess.Popen(args, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)


def start_ffmpeg_process2(out_filename, width, height,fps):
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height),r=fps,hwaccel=hwaccel,loglevel="quiet").video
        .output(out_filename, movflags='faststart',pix_fmt='yuv420p',vcodec=codec,video_bitrate='11M',preset='slow',loglevel="quiet")
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)

def read_frame(process1, width, height):
    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame


def predict_frame(in_frame,w,h,model,score,device,rect_op,copyright,name_color,fixwin_color):

    out_frame=in_frame.copy()
    rsz_frame=in_frame.copy()

    rsz_frame=cv2.resize(rsz_frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)

    results = model.predict(source=rsz_frame,conf=score,device=device,imgsz=int(w*0.5),show_labels=False,show_conf=False,show_boxes=False)

    if len(results[0]) > 0:
        for box in results[0].boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])  # バウンディングボックスの座標
            out_frame = cv2.rectangle(out_frame ,(int(xmin*(1/0.5)), int(ymin*(1/0.5))),(int(xmax*(1/0.5)), int(ymax*(1/0.5))),name_color,-1)
    if copyright==True:
        out_frame=put_C_SQUARE_ENIX(out_frame,w,h)
    for box_op in rect_op:
        xmin_op, ymin_op, xmax_op, ymax_op = map(int, box_op)
        out_frame = cv2.rectangle(out_frame ,(xmin_op, ymin_op),(xmax_op, ymax_op),fixwin_color,-1)

    return out_frame

def write_frame(process2, frame):
    process2.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )
def put_C_SQUARE_ENIX(img,w,h):
    dy=h-41
    dx=w-220
    image_h, image_w = 31,200
    img[dy:dy+image_h, dx:dx+image_w] = c_sqex_image
    return img

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

def main(page: ft.Page):

    logger = getLogger('ultralytics')
    logger.disabled = True

    f_score_init=0.20

    rect_op=[]

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
    def video_main(in_filename, out_filename,score:float, process_frame,start_time,end_time,copyright:bool,name_color,fixwin_color):

        elapsed_i=0
        trim_skip=False

        process_state=0 #output flag

        video_temp_filename_1 = 'tmp_wol_'+str(int(time.time()))+'_1.mp4'
        video_temp_filename_2 = 'tmp_wol_'+str(int(time.time()))+'_2.mp4'
        audio_temp_filename = 'tmp_wol_'+str(int(time.time()))+'.m4a'

        width, height,vfps,frame_max ,color_primaries= get_video_size(in_filename)

        all_sec=int(frame_max//vfps)
        mod_frame=float(frame_max%vfps)
        if mod_frame>0:
            mod_frame_sec=1
        else:
            mod_frame_sec=0
        all_sec=all_sec+mod_frame_sec

        if start_time != 0 or end_time != all_sec:
            snack_bar_message("Video Trimming...")
            page.add(ffmpeg_info_text)

            stream = (
                ffmpeg
                .input(in_filename, ss=start_time, t=end_time-start_time,hwaccel=hwaccel)
                .output(video_temp_filename_1,vcodec='copy',acodec='copy',format='mp4',video_bitrate='11M',preset='slow')
                .overwrite_output()
                .compile()
                )
            process0=subprocess.Popen(stream, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,universal_newlines=True)
            for line in process0.stdout:
                ffmpeg_info_text.value=line
                ffmpeg_info_text.update()
            process0.wait()
            page.remove(ffmpeg_info_text)
        else:
            trim_skip=True

        snack_bar_message("WolNamesBlackedOut...")

        current_frame_number = 0  # 初期値
        start = time.time()

        if trim_skip==True:
            video_1=in_filename
        else:
            video_1=video_temp_filename_1

        process1 = start_ffmpeg_process1(video_1,color_primaries)
        process2 = start_ffmpeg_process2(video_temp_filename_2, width, height,vfps)

        while True:
            in_frame = read_frame(process1, width, height)
            if in_frame is None:
                process_state=1
                break

            current_frame_number += 1

            out_frame = process_frame(in_frame,width, height,model,score,0,rect_op,copyright,name_color,fixwin_color)
            write_frame(process2, out_frame)

            elapsed_i=time.time()-start
            frame_progress.value=str(int(current_frame_number))+'/'+str(vfps*(end_time-start_time))
            elapsed.value=str(format(elapsed_i,'.2f'))+'s'
            fps.value = str(format(int(current_frame_number) / elapsed_i,'.2f'))
            percentage = int(current_frame_number)/(vfps*(end_time-start_time))
            eta.value = str(int(elapsed_i * (1 - percentage) / percentage + 0.5))+'s'
            pb.value=percentage
            pb.update()
            elapsed.update()
            fps.update()
            eta.update()
            frame_progress.update()

            if start_button.disabled == False:
                process_state=-1    #output cancel
                process1.kill()
                process2.kill()
                break

        process1.wait()

        process2.stdin.close()
        process2.wait()

        if process_state==1:
            snack_bar_message("Video process Finished. Audio process Start")
        elif process_state==-1:
            snack_bar_message("Video process Stopped.")

        page.add(ffmpeg_info_text)

        if process_state==1:
            # audio track output
            page.add(image_ring)

            audio=(
                ffmpeg
                .input(in_filename)
                .audio
                )
            stream=ffmpeg.output(audio,audio_temp_filename,acodec='copy').overwrite_output().compile()
            p1=subprocess.Popen(stream, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,universal_newlines=True)
            for line in p1.stdout:
                ffmpeg_info_text.value=line
                ffmpeg_info_text.update()
            try:
                result = p1.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            else:
                p1.kill()

            input_video=ffmpeg.input(video_temp_filename_2)
            if trim_skip==True:
                input_audio=(
                    ffmpeg.input(audio_temp_filename)
                    )
                stream=ffmpeg.output( input_audio, input_video.video,out_filename,movflags='faststart',acodec='copy', vcodec='copy',format='mp4').overwrite_output().compile()
            else:
                input_audio=(
                    ffmpeg.input(audio_temp_filename)
                    .filter('atrim', start=start_time, end=end_time,)
                    .filter('asetpts', 'PTS-STARTPTS')
                    )
                stream=ffmpeg.output( input_audio, input_video.video,out_filename,movflags='faststart', vcodec=codec,format='mp4',video_bitrate='11M',preset='slow').overwrite_output().compile()

            p2=subprocess.Popen(stream, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,universal_newlines=True)

            for line in p2.stdout:
                ffmpeg_info_text.value=line
                ffmpeg_info_text.update()
            page.remove(ffmpeg_info_text)

            try:
                result = p2.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            else:
                p2.kill()

            page.remove(image_ring)

            snack_bar_message("Movie Complete")

        if trim_skip==False:
            os.remove(video_temp_filename_1)
        os.remove(video_temp_filename_2)
        if process_state==1:
            os.remove(audio_temp_filename)

        process_finished()

    def image_main(video_in:str,frame:int,score:float,copyright:bool,name_color:str,fixwin_color:str):
        name_color_r=int(name_color[1:3],base=16)
        name_color_g=int(name_color[3:5],base=16)
        name_color_b=int(name_color[5:],base=16)
        fixwin_color_r=int(fixwin_color[1:3],base=16)
        fixwin_color_g=int(fixwin_color[3:5],base=16)
        fixwin_color_b=int(fixwin_color[5:],base=16)

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
                    cv2.rectangle(img_tmp,(x1,y1),(x,y),(fixwin_color_b,fixwin_color_g,fixwin_color_r),-1)
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

        cap = cv2.VideoCapture(video_in)
        width, height,vfps,frame_max ,color_primaries= get_video_size(video_in)

        if not cap.isOpened():
            return

        w = width
        h = height

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

        ret, frame = cap.read()

        if color_primaries=='bt2020':
            frame=apply_tone_mapping(frame)

        rsz_frame=frame.copy()
        rsz_frame=cv2.resize(rsz_frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)

        results = model.predict(source=rsz_frame,conf=score,device=0,imgsz=int(w*0.5),show_labels=False,show_conf=False,show_boxes=False)

        # Display the annotated frame
        if len(results[0]) > 0:
            for box in results[0].boxes:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])  # bbox
                frame = cv2.rectangle(frame ,(int(xmin*(1/0.5)), int(ymin*(1/0.5))),(int(xmax*(1/0.5)), int(ymax*(1/0.5))),(name_color_b, name_color_g, name_color_r),-1)
        if copyright==True:
            frame=put_C_SQUARE_ENIX(frame,w,h)
        frame_op_add=frame.copy()
        for box_op in rect_op:
            xmin, ymin, xmax, ymax = map(int, box_op)
            frame_op_add = cv2.rectangle(frame_op_add ,(xmin, ymin),(xmax, ymax),(fixwin_color_b,fixwin_color_g, fixwin_color_r),-1)

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
        preview_button.disabled=False
        preview_button.icon_color=ft.colors.GREEN
        pb.value=0
        frame_progress.value='0000/0000'
        check_copyright.disabled=False
        score_threshold_slider.disabled=False
        save_file_path.disabled=False
        selected_files.disabled=False
        resize_slider.disabled=False
        video_frame_slider.disabled=False
        pcname_color.disabled=False
        fixwindow_color.disabled=False
        page.update()

    def start_clicked(e):
        e.control.disabled = True
        e.control.icon_color=''
        preview_button.disabled = True
        preview_button.icon_color=''
        stop_button.disabled = False
        stop_button.icon_color=ft.colors.RED
        check_copyright.disabled=True
        score_threshold_slider.disabled=True
        save_file_path.disabled=True
        selected_files.disabled=True
        resize_slider.disabled=True
        video_frame_slider.disabled=True
        pcname_color.disabled=True
        fixwindow_color.disabled=True
        page.update()
        name_color_r=int(str(pcname_color.icon_color)[1:3],base=16)
        name_color_g=int(str(pcname_color.icon_color)[3:5],base=16)
        name_color_b=int(str(pcname_color.icon_color)[5:],base=16)
        fixwin_color_r=int(str(fixwindow_color.icon_color)[1:3],base=16)
        fixwin_color_g=int(str(fixwindow_color.icon_color)[3:5],base=16)
        fixwin_color_b=int(str(fixwindow_color.icon_color)[5:],base=16)
        pcname_color_bgr=(name_color_b,name_color_g,name_color_r)
        fixwin_color_bgr=(fixwin_color_b,fixwin_color_g,fixwin_color_r)
        video_main(selected_files.value,save_file_path.value,float(slider_t.value),predict_frame, int(frame_range_slider_start_min.value)*60+int(frame_range_slider_start_sec.value) , int(frame_range_slider_end_min.value)*60+int(frame_range_slider_end_sec.value),check_copyright.value,pcname_color_bgr,fixwin_color_bgr)


    def stop_clicked(e):
        e.control.disabled = True
        e.control.icon_color=''
        start_button.disabled = False
        start_button.icon_color=ft.colors.BLUE
        preview_button.disabled=False
        preview_button.icon_color=ft.colors.GREEN
        pcname_color.disabled=False
        fixwindow_color.disabled=False
        page.update()

    def preview_clicked(e):
        preview_button.disabled = True
        preview_button.icon_color=''
        start_button.disabled = True
        start_button.icon_color=''
        pcname_color.disabled=True
        fixwindow_color.disabled=True
        page.add(image_ring)
        page.update()
        image_main(selected_files.value,int(frame_slider_t.value),float(slider_t.value),check_copyright.value,str(pcname_color.icon_color),str(fixwindow_color.icon_color))
        page.remove(image_ring)
        preview_button.disabled=False
        preview_button.icon_color=ft.colors.GREEN
        start_button.disabled = False
        start_button.icon_color=ft.colors.BLUE
        pcname_color.disabled=False
        fixwindow_color.disabled=False
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

    frame_slider_t = ft.TextField(label='Frame',value=0,width=80,input_filter=ft.NumbersOnlyInputFilter(),on_change=frame_textfield_change)
    frame_max_t=ft.Text(value='/ 0')

    slider_t = ft.Text(value=f_score_init)

    def pick_files_result(e: ft.FilePickerResultEvent):
        nonlocal rect_op
        selected_files.value = (
            ", ".join(map(lambda f: f.name, e.files)) if e.files else "Cancelled!"
        )
        if e.files or selected_files.value != "Cancelled!":
            selected_files.value=e.files[0].path
            width, height,vfps,frame_max ,color_primaries= get_video_size(selected_files.value)
            cap = cv2.VideoCapture(selected_files.value)
            video_frame_slider.max=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frame_slider.divisions=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frame_slider.disabled=False
            video_frame_slider.value=0
            video_frame_slider.update()
            all_sec=int(frame_max//vfps)
            mod_frame=float(frame_max%vfps)
            if mod_frame>0:
                mod_frame_sec=1
            else:
                mod_frame_sec=0
            all_min=int(all_sec//60)
            mod_sec=int(all_sec-all_min*60)+mod_frame_sec
            if mod_sec==60:
                all_min=all_min+1
                mod_sec=0
            frame_range_slider_start_min.value=0
            frame_range_slider_start_sec.value=0
            frame_range_slider_start_sec.update()
            frame_range_slider_start_min.update()
            frame_range_slider_end_min.value=all_min
            frame_range_slider_end_sec.value=mod_sec
            frame_range_slider_end_sec.update()
            frame_range_slider_end_min.update()
            frame_slider_t.value=0
            frame_slider_t.update()
            frame_max_t.value='/ '+str(frame_max)
            frame_max_t.update()
            resize_slider.disabled=False
            resize_slider.update()
            start_button.disabled=False
            start_button.update()
            preview_button.disabled=False
            preview_button.update()
            fps.value='0.00s'
            eta.value='0.00s'
            elapsed.value='0.00s'
            frame_progress.value='0000/0000'
            fps.update()
            eta.update()
            elapsed.update()
            cap.release()
            rect_op=[]
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

    def frame_range_start_min_change(e):
        if e.control.value=='':
            e.control.value=0
            e.control.update()
        if int(frame_range_slider_start_min.value)*60+int(frame_range_slider_start_sec.value) > int(frame_range_slider_end_min.value)*60+int(frame_range_slider_end_sec.value):
            start_button.disabled=True
        else:
            start_button.disabled=False
        start_button.update()

    def frame_range_start_sec_change(e):
        if e.control.value=='':
            e.control.value=0
            e.control.update()
        if int(frame_range_slider_start_min.value)*60+int(frame_range_slider_start_sec.value) > int(frame_range_slider_end_min.value)*60+int(frame_range_slider_end_sec.value):
            start_button.disabled=True
        else:
            start_button.disabled=False
        start_button.update()

    def frame_range_end_min_change(e):
        if e.control.value=='':
            e.control.value=0
            e.control.update()
        if int(frame_range_slider_start_min.value)*60+int(frame_range_slider_start_sec.value) > int(frame_range_slider_end_min.value)*60+int(frame_range_slider_end_sec.value):
            start_button.disabled=True
        else:
            start_button.disabled=False
        start_button.update()

    def frame_range_end_sec_change(e):
        if e.control.value=='':
            e.control.value=0
            e.control.update()
        if int(frame_range_slider_start_min.value)*60+int(frame_range_slider_start_sec.value) > int(frame_range_slider_end_min.value)*60+int(frame_range_slider_end_sec.value):
            start_button.disabled=True
        else:
            start_button.disabled=False
        start_button.update()

    def snack_bar_message(e):
        page.snack_bar = ft.SnackBar(ft.Text(e))
        page.snack_bar.open = True
        page.update()


    update_check()
    page.overlay.extend([pick_files_dialog, save_file_dialog])

    score_threshold_slider=ft.Slider(min=0, max=0.5, value=f_score_init,divisions=50,on_change=slider_change)
    check_copyright=ft.Checkbox(label="Add Copyright", value=True)

    video_frame_slider=ft.Slider(min=0, max=1000, divisions=1000,width=300,disabled=True,on_change=frame_slider_change)

    image_ring=ft.ProgressBar(color=ft.colors.LIGHT_BLUE_400)
    ffmpeg_info_text=ft.Text(value=' ')

    resize_slider=ft.Slider(value=80,min=50, max=100,width=120, divisions=5, disabled=True,label="Resize {value}%")
    frame_range_slider_start_min = ft.TextField(label='Start_min',value=0,width=80,read_only=False,input_filter=ft.NumbersOnlyInputFilter(),on_change=frame_range_start_min_change)
    frame_range_slider_start_sec = ft.TextField(label='Start_sec',value=0,width=80,read_only=False,input_filter=ft.NumbersOnlyInputFilter(),on_change=frame_range_start_sec_change)
    frame_range_slider_end_min = ft.TextField(label='End_min',value=0,width=80,read_only=False,input_filter=ft.NumbersOnlyInputFilter(),on_change=frame_range_end_min_change)
    frame_range_slider_end_sec = ft.TextField(label='End_sec',value=0,width=80,read_only=False,input_filter=ft.NumbersOnlyInputFilter(),on_change=frame_range_end_sec_change)

    def pcname_color_icon():

        async def open_color_picker(e):
            e.control.page.dialog = d
            d.open = True
            e.control.page.update()

        color_picker = ColorPicker(color="#c8df6f", width=300)
        color_button = ft.ElevatedButton("BlackedOut_color",icon=ft.icons.BRUSH,icon_color='#000000',on_click=open_color_picker)

        async def change_color(e):
            color_button.icon_color = color_picker.color
            d.open = False
            e.control.page.update()

        async def close_dialog(e):
            d.open = False
            d.update()

        d = ft.AlertDialog(
            content=color_picker,
            actions=[
                ft.TextButton("OK", on_click=change_color),
                ft.TextButton("Cancel", on_click=close_dialog),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            on_dismiss=change_color,
        )

        return color_button

    def fixwindow_color_icon():

        async def open_color_picker(e):
            e.control.page.dialog = d
            d.open = True
            e.control.page.update()

        color_picker = ColorPicker(color="#c8df6f", width=300)
        color_button = ft.ElevatedButton("FixWindow_color",icon=ft.icons.BRUSH,icon_color='#000000',on_click=open_color_picker)

        async def change_color(e):
            color_button.icon_color = color_picker.color
            d.open = False
            e.control.page.update()

        async def close_dialog(e):
            d.open = False
            d.update()

        d = ft.AlertDialog(
            content=color_picker,
            actions=[
                ft.TextButton("OK", on_click=change_color),
                ft.TextButton("Cancel", on_click=close_dialog),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            on_dismiss=change_color,
        )

        return color_button
    pcname_color = pcname_color_icon()
    fixwindow_color = fixwindow_color_icon()

    page.padding=10
    page.window.width=700
    page.window.height=830
    page.window.title_bar_hidden = True
    page.window.title_bar_buttons_hidden = True
    page.window.icon=resource_path("WoLNamesBlackedOut.ico")

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
                        file_type=ft.FilePickerFileType.VIDEO,
                        allowed_extensions=["mp4"]
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
                # ft.Text("  "),
                ft.Text("Score Threshold"),
                score_threshold_slider,
                slider_t,
                ft.Text("  "),
                check_copyright,
                ]
        ),
        ft.Row(controls=[
                pcname_color,
                fixwindow_color,
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
                ft.Text(" Video frame"),
                video_frame_slider,
                frame_slider_t,
                frame_max_t,
                ]
        ),
        ft.Row(controls=
            [
                preview_button,
                resize_slider,
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
                frame_range_slider_start_min,
                ft.Text(":"),
                frame_range_slider_start_sec,
                ft.Text(" - "),
                frame_range_slider_end_min,
                ft.Text(":"),
                frame_range_slider_end_sec,
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