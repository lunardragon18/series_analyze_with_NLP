import gradio as gr
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from theme_classifier.theme import Theme

def get_themes(theme_list,subtitles_path,save_path):
    theme_list = theme_list.split(',')
    themes = Theme(theme_list)
    df = themes.save_themes(subtitles_path,save_path)
    theme_list = [theme for theme in theme_list if theme != "dialogue"]
    output = df[theme_list].sum().reset_index()
    output.columns = ['Theme','Score']

    plot = gr.BarPlot(output,x="Theme",y="Score",title="Themes", tooltip=["Theme","Score"],vertical=False,width=500,height=250)
    return plot

def main():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classifier (Zero Shot Classifier)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label="Theme_box")
                        subtitles_path = gr.Textbox(label ="Subtitles path")
                        save_path = gr.Textbox(label="Save_path")
                        get_themes_b = gr.Button("Get_Themes")
                        get_themes_b.click(get_themes, inputs=[theme_list,subtitles_path,save_path],outputs=[plot])




    demo.launch(share = True)

if __name__ ==  "__main__":
    main()

